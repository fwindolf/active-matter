import numpy as np
import cv2
import os
import glob

import queue
import threading
import logging
import argparse

logging.basicConfig(format='%(thread)d | %(levelname)s | %(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

IMG_SHAPE = (1600, 1600)

def load_img(image_name):
    log.debug("Loading image from file %s" % image_name)
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return image

def threshold(image, threshold=150):
    thresholded = image.copy()
    thresholded[image >= 255 - threshold] = 255
    return thresholded

def extract_line(image, threshold=100):
    log.debug("Extracting line")
    # opening with elliptic kernel
    inverted_image = (image < threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 9))
    image = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel)
  
    # dilation with rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    line_image = cv2.dilate(image, kernel, iterations=1)
    
    # interpret all the line as points and fit polynomial
    points = cv2.findNonZero((line_image > 0.5).astype(np.uint8))
    x = [p[0][0] for p in points]
    y = [p[0][1] for p in points]
    poly = np.poly1d(np.polyfit(x, y, 2))
    log.debug("Line found with parameters: %s" % str(poly.c))
    
    return poly

def undistort_by_shifting(image, line):
    xl = np.arange(0, image.shape[1])
    y = line(xl)
    y_min = np.min(y)
    y_shift = (y - y_min).astype(np.uint8)
    
    # cut into pieces and roll
    parts = list()
    
    idx = 0
    idx_start = 0
    last_s = y_shift[0]
    for idx, s in enumerate(y_shift):
        # start new chunk whenever the amount of pixels to shift changes
        if idx < len(y_shift) or y_shift[idx] != last_s:            
            # get the chunk
            part = image[:, idx_start:idx]
            if last_s != 0:
                # roll upwards if there is something to roll
                part = np.roll(part, part.shape[0]-last_s, axis=0)
            
            parts.append(part)
            idx_start = idx
            
        last_s = s
       
    # stitch parts back together
    im = np.concatenate(parts, axis=1)    
    log.debug("Undistorted image")
    
    # create a new line
    newline = np.poly1d([y_min])
    log.debug("New line after undistorting: %s" % str(newline.c))
    
    return im, newline

def cut_over_line(image, line, offset=2):
    xl = np.arange(image.shape[1])
    y_cut = int(np.min(line(xl)))
    im = image.copy()
    im = im[:y_cut + offset, ...]

    if im.shape[0] < IMG_SHAPE[0] or im.shape[1] < IMG_SHAPE[1]:
        raise AttributeError("Image too small!")        

    # cut to IMG_SHAPE, so we lose image at the sides and the top
    d = im.shape[1] - IMG_SHAPE[1]
    dl = int(d/2)
    dr = d-dl
    im = im[max(0, im.shape[0] - IMG_SHAPE[0]):, dl:im.shape[1]-dr]
    return im

def get_templates(r_template, n_templates=5, name='template*.png'):
    template_files = sorted(glob.glob("templates/" + name))
    templates = []
    for i, t in enumerate(template_files):
        img = cv2.imread(t)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (2*r_template, 2*r_template))
        img = threshold(img)
        
        templates.append(img)
        if i > n_templates:
            break
    
    log.debug("Created %d templates for template matching" % len(templates))
    return templates

def get_activations(image, suppression=0.55, r_template=10):
    activations = []
    for t in get_templates(r_template, name='template*.png'):
        activation = cv2.matchTemplate(image, t, cv2.TM_SQDIFF_NORMED) 
        activations.append(activation)
        
    # extract activations that are between median and max activation
    average_activations = 1. - np.mean(activations, axis=0)
    max_activation = np.max(average_activations)    
    final_activations = (average_activations >= suppression * max_activation).astype(np.uint8)

    log.debug("Found activations for image")
    return final_activations


def limit_to_area_in_particles(image, activations):
    log.debug("Reducing activations to areas where there are particles")
    # image is thresholded, use it in a way that the particles are active
    img = np.invert(image)
    img = img.astype(np.bool).astype(np.uint8) # binarize

    # get rid of the insides of particles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # make the particle area slightly smaller
    img = cv2.erode(img, kernel, iterations=2)
    
    # align the images
    ox = img.shape[0] - activations.shape[0]
    oxr = int(ox/2)
    oxl = ox - oxr
    oy = img.shape[1] - activations.shape[1]
    oyt = int(oy/2)
    oyb = oy - oyt    
    mask = img[oxl:-oxr, oyt:-oyb]

    # only use the activations where it is inside the particles
    final_activations = activations.copy()
    final_activations[mask == 0] = 0

    log.debug("Activations pruned")
    return final_activations

def find_point_candidates(activations, r_template):
    _, contours, _ = cv2.findContours(activations.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    log.debug("Found %d contours that could be point candidates" % len(contours))
    # extract the points from good looking contours
    points = []
    for cnt in contours:
        # sort out the contours that are too big/too small
        A = cv2.contourArea(cnt)
        if A < 2 or A > 100:
            continue
        
        M = cv2.moments(cnt)
        # offset by r_template bc that's what we lose during template matching
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        points.append((cX, cY))
    
    log.debug("%d point candidates found" % len(points))
    return points  

def sqdist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def ldist(p, polinomial):
    p1 = np.asarray((p[0] - 10, polinomial(p[0] - 10)))
    p2 = np.asarray((p[0] + 10, polinomial(p[0] + 10)))
    p3 = np.asarray(p)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3))/ np.linalg.norm(p2 - p1)    

def is_critical(sqdist, r_template):
    radius_crit = 1.7* r_template
    return sqdist < radius_crit

def is_neighbor(sqdist, r_template):
    radius_neig = 2.3 * r_template
    return sqdist < radius_neig

def is_extended_neigbor(sqdist, r_template):
    ext_radius_neig = 2.8 * r_template
    return sqdist < ext_radius_neig

def find_neighbors(points, save_points=set(), crit_points=set(), r_template=10):
    neighbors = list()
    log.debug("Searching for neighbors")
    
    # TODO: Speed up via quadtrees possible
    for i, p in enumerate(points):
        # check if it interferes with another point
        overlaps = 0    
        this_neighbors = []
        for j, p_check in enumerate(points): 
            if i == j:
                continue

            d = sqdist(p, p_check)
            if is_critical(d, r_template):
                overlaps += 1

            if is_neighbor(d, r_template):
                this_neighbors.append(j)

        if overlaps < 1:
            save_points.add(i)

        if overlaps > 2:
            crit_points.add(i)

        neighbors.append(this_neighbors)
    
    log.debug("Neighbors for all points saved")
    return neighbors

def find_line_intersecting_points(points, line, r_template):
    # find every point that interferes with the line to remove later
    to_delete = set()
    log.debug("Deleting point candidates that intersect with the boundary")
    for i, p in enumerate(points):
        if ldist(p, line) < r_template:
            to_delete.add(i)   
    
    return to_delete

def find_normal_intersecting_points(points, neighbors, to_delete=set(), save_points=set(), crit_points=set(), r_template=10):
    # make every normal point that intersects with another normal point a critical one
    log.debug("Making point candidates critical when they intersect with normal points")
    for i, p in enumerate(points):
        if i in (to_delete | crit_points | save_points):
            continue

        for n in neighbors[i]:
            if n in (to_delete | crit_points | save_points):
                continue

            d = sqdist(p, points[n])
            if is_critical(d, r_template):
                crit_points.add(i)
                crit_points.add(n)
                
    return crit_points

def convert_normal_to_save(points, neighbors, to_delete, save_points, crit_points, r_template):
    change = False
    log.debug("Converting point candidate to save by counting the number of save neighbors")
    # convert non-critical points to save if they have enough save neighbors
    for i, p in enumerate(points):
        # skip all special points
        if i in (to_delete | crit_points | save_points):
            continue
        
        # convert a point if it is connected to a neighbor, but doesnt intersect
        save_n = 0
        for n in neighbors[i]:
            if n in (to_delete | crit_points) or n not in save_points:
                continue 

            d = sqdist(p, points[n])
            if is_critical(d, r_template):
                # it intersects, so delete p as it intersects with save point
                change = True
                crit_points.add(i)
            else:
                # it is neighbor
                save_n += 1
                
        if save_n >= 2:
            change = True
            save_points.add(i)

    log.debug("Stats: %d points, %d to_delete, %d save_points, %d crit_points" % (len(points), len(to_delete), len(save_points), len(crit_points)))
    log.debug("Convert_normal_to_save: Change is %r" % change)          
    return change

def delete_critical_points_intersecting_save(points, neighbors, to_delete, save_points, crit_points, r_template):
    # delete all critical points that intersect with save points
    log.debug("Deleting point candidates that overlap with save points")                    
    change = False
    for i in save_points:
        for n in neighbors[i]:
            if n not in crit_points:
                continue

            d = sqdist(points[i], points[n])
            if is_critical(d, r_template):
                change = True
                to_delete.add(n)

    log.debug("Stats: %d points, %d to_delete, %d save_points, %d crit_points" % (len(points), len(to_delete), len(save_points), len(crit_points)))
    log.debug("delete_critical_points_intersecting_save: Change is %r" % change)          
    return change

def convert_critical_to_normal(points, neighbors, to_delete, save_points, crit_points, r_template):
    # make all critical points normal points that dont intersect with a normal/save point
    change = False
    log.debug("Converting critical points with no overlap to other candidates to normal")                    

    for i, p in enumerate(points):
        if i not in crit_points: # cannot iterate crit_points as we modify 
            continue

        n_crit = 0
        for n in neighbors[i]:
            if n in (crit_points | to_delete):
                continue

            d = sqdist(p, points[n])
            if is_critical(d, r_template):
                n_crit += 1

        if n_crit == 0:        
            change = True
            crit_points.discard(i)

    log.debug("Stats: %d points, %d to_delete, %d save_points, %d crit_points" % (len(points), len(to_delete), len(save_points), len(crit_points)))
    log.debug("convert_critical_to_normal: Change is %r" % change)                    
    return change

def iteratively_propagate_safeness(points, neighbors, to_delete, save_points, crit_points, r_template):
    change = True
    log.debug("Propagating safeness of points")
    log.debug("Stats: %d points, %d to_delete, %d save_points, %d crit_points" % (len(points), len(to_delete), len(save_points), len(crit_points)))
    while(change):
        change = False
        
        change |= convert_normal_to_save(points, neighbors, to_delete, save_points, crit_points, r_template) 
        
        change |= delete_critical_points_intersecting_save(points, neighbors, to_delete, save_points, crit_points, r_template)
        
        change |= convert_critical_to_normal(points, neighbors, to_delete, save_points, crit_points, r_template)

        # remove deleted points from crit_points
        crit_points = crit_points - to_delete
        log.debug("Stats: %d points, %d to_delete, %d save_points, %d crit_points" % (len(points), len(to_delete), len(save_points), len(crit_points)))

def prune_point_candidates(points, line, r_template):
    save_points = set()
    crit_points = set()
    to_delete = set()

    neighbors = find_neighbors(points, save_points, crit_points, r_template)
    
    to_delete |= find_line_intersecting_points(points, line, r_template)
    
    crit_points |= find_normal_intersecting_points(points, neighbors, to_delete, save_points, crit_points, r_template)
       
    iteratively_propagate_safeness(points, neighbors, to_delete, save_points, crit_points, r_template)
    
    found_points = [p for i, p in enumerate(points) if i not in (to_delete | crit_points)]
    return found_points

def write_image(image, path, name, image_folder):
    file_dir = os.path.join(path, image_folder) + "/" + name + ".jpg"
    log.debug("Writing output image to %s" % file_dir)
    cv2.imwrite(file_dir, image)

def write_particles(points, path, name, txt_folder):
    file_dir = os.path.join(path, txt_folder) + "/" + name + ".xyz"
    log.debug("Writing %d particle positions to %s" % (len(points), file_dir))
    with open(file_dir, 'w') as f:
        f.write("%d\n" % len(points))
        f.write("i x y z l\n")

        for i, p in enumerate(points):
            f.write("%d\t%.6f\t%.6f\t%.6f\t%d\n" % (i, p[0], p[1], 0, 0))

def extract_particles(i, image, directory):
    # load and extract boundary
    img = load_img(image)
    line = extract_line(threshold(img))

    # remove distortion (flatten along boundary)
    img_shifted, line_shifted = undistort_by_shifting(img, line)
    img_thresholded_shifted = threshold(img_shifted)

    # only use part above boundary
    img_cut = cut_over_line(img_shifted, line_shifted)
    img_cut_thresholded = cut_over_line(img_thresholded_shifted, line_shifted)

    # find activations for particles by template matching (and pruning)
    img_activation = limit_to_area_in_particles(img_cut_thresholded, get_activations(img_cut_thresholded, r_template=args.radius))

    # extract points by looking at contours of activations
    points = find_point_candidates(img_activation, args.radius)

    # prune out invalid candidates
    extracted_points = prune_point_candidates(points, line_shifted, args.radius)

    # save to disk
    name = "%05d" % i
    write_image(img_cut, directory, name, 'images')
    write_particles(extracted_points, directory, name, 'txt')

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Extract particles from unprocessed images')

    parser.add_argument('-d', '--data_path', 
                        help='Path to the original data')
    parser.add_argument('-o', '--output_path', 
                        help='Path where the output data should be placed')
    parser.add_argument('-r', '--radius', type=int, default=10,
                        help='Radius of the particles in the images in pixels')
    parser.add_argument('-n', '--num_threads', type=int, default=4,
                        help='Number of threads used to process images')
    parser.add_argument('-l', '--log_level', choices=['FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'], default='INFO',
                        help='Level for logging output')
    
    args = parser.parse_args()

    # Set logging level from string
    log.setLevel(logging.getLevelName(args.log_level))
    
    q = queue.Queue(args.num_threads)

    log.info("Extracting particles from %s" % args.data_path)

    def work():
        while True:
            item = q.get()
            if item is None:            
                log.debug("Last item received")
                q.task_done()
                break
            
            i, image, directory = item
            log.debug("Got new item from queue")
            
            extract_particles(i, image, directory)

            q.task_done()

        log.debug("Stopping thread %d" % threading.get_ident())

    threads = []
    log.info("Creating %d worker threads" % args.num_threads)
    for i in range(args.num_threads):
        t = threading.Thread(target=work)
        t.start()
        threads.append(t)

    for folder in os.listdir(args.data_path):        
        images = sorted(glob.glob(os.path.join(args.data_path, folder) + "/*.tif"))
        log.info("Found folder %s in path %s with %d images" % (folder, args.data_path, len(images)))
        directory = os.path.join(args.output_path, folder)
        
        log.info("Saving to %s/txt,/images" % directory)
        os.makedirs(os.path.join(directory, 'txt'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'images'), exist_ok=True)

        for i, img_name in enumerate(images):
            log.debug("Putting new work item into queue")
            if i % 100 == 0:
                log.info("%d/%d images" % (i, len(images)))
            q.put((i, img_name, directory))

    log.info("Finished enqueuing items")
    q.join()

    log.info("Stopping and joining threads")
    for _ in threads:
        q.put(None)

    for t in threads:
        t.join()
    
    log.info("Done")