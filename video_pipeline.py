import argparse
from universal_datagen.generator.generator_text import AM2018TxtGenerator
from universal_models.models.models import get_model

import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)
import keras
from keras.models import model_from_json, Model
from keras.layers import Input

import os
import glob
import json
import time
import queue
import threading
import logging

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) # like tf

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('-am', '--abstraction_model_dir', help='Path to the model dir for image abstraction (includes model.json, param.json, weights file)', required=True)
parser.add_argument('-cm', '--classification_model_dir', help='Path to the model dir for classification (includes model.json, param.json, weights file)', required=True)
parser.add_argument('-d', '--data', help='Video name or deviceId for input data', required=True)
parser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess the input data')

parser.add_argument('-f', '--fps', type=float, default=20.0, help='FPS for the video')
parser.add_argument('-o', '--output_name', default='output', help='Name of the video')
parser.add_argument('-b', '--batch_size', help='Speed up by increasing the number of frames fed to the model', type=int, default=1)
parser.add_argument('-m', '--max_frames', help='Maximum number of frames to process', type=int, default=100000)
parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode (only show video)')

args = parser.parse_args()

with open(os.path.join(args.abstraction_model_dir, 'param.json'), 'r') as f:
    param_dict = json.loads(f.read())
    ih = param_dict['dataset_input_height']
    iw = param_dict['dataset_input_width']
    ic = param_dict['dataset_input_channels']
    icl = param_dict['dataset_num_classes']

with open(os.path.join(args.classification_model_dir, 'param.json'), 'r') as f:
    param_dict = json.loads(f.read())
    lh = param_dict['dataset_input_height']
    lw = param_dict['dataset_input_width']
    lc = param_dict['dataset_input_channels']
    lcl = param_dict['dataset_num_classes']

def fig_to_np(fig):
    # Render into numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    im = np.reshape(im, (h, w, 3)) # rgb
    im = np.stack((im[..., 2], im[..., 1], im[..., 0],), axis=2) # bgr

    return im

# Initialize image shape
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
im_x, im_y = axes.ravel()

x = np.zeros((ih, iw))
y = np.zeros((lh, lw))

im_x.set_axis_off()
im_x.imshow(x, cmap='hot')
im_x.set_title("Input")

im_y.set_axis_off()
im_y.imshow(y, cmap='hot')
im_y.set_title("Prediction")

plt.tight_layout()

im = fig_to_np(fig)
h, w, _ = im.shape

if args.interactive:
    cv2.namedWindow("Output")
else:
    print("Creating video writer")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('videos/' + args.output_name + '.mp4',fourcc, args.fps, (w, h)) # w, h

    # Test video writer
    try:
        out.write(im)
    except cv2.error as e:
        logging.critical("Tested to write empty frame to video but failed. This might indicate wrong image/video dimensions")
        exit(0)

    # renew to get rid of blank frame
    out = cv2.VideoWriter('videos/' + args.output_name + '.mp4',fourcc, args.fps, (w, h)) # w, h

print("Creating video capture")
inp = cv2.VideoCapture()
status = inp.open(args.data)
if not status:
    raise RuntimeError("Unable to open video (stream) %s" % args.data)

fq = queue.Queue(30)
vq = queue.Queue(30)

def feed_model():
    # create model in this thread
    logging.debug("Loading model file %s" % (args.abstraction_model_dir + 'model.json'))
    with open(args.abstraction_model_dir + '/model.json', 'r') as f:
        abstract_model = model_from_json(f.read())

    weights = glob.glob(args.abstraction_model_dir + "weights*.hdf5")
    logging.debug("Loading weights file %s" % (sorted(weights)[-1]))
    abstract_model.load_weights(sorted(weights)[-1])

    new_input = Input(shape=(1, ih, iw, ic)) # time step of 1       
    abstract_model.layers.pop(0)
    new_output = abstract_model(new_input)

    abstract_model = Model(new_input, new_output)

    logging.debug("Loading model file %s" % (args.classification_model_dir + 'model.json'))
    with open(args.classification_model_dir + 'model.json', 'r') as f:
        classify_model = model_from_json(f.read())
    
    weights = glob.glob(args.classification_model_dir + "weights*.hdf5")
    logging.debug("Loading weights file %s" % (sorted(weights)[-1]))
    classify_model.load_weights(sorted(weights)[-1])
                        
    new_input = Input(shape=(1, lh, lw, lc)) # time step of 1
    classify_model.layers.pop(0)
    new_output = classify_model(new_input)

    classify_model = Model(new_input, new_output)

    logging.debug("Classification model created")

    pipeline_model = Model(abstract_model.inputs, classify_model(abstract_model(abstract_model.inputs)))
    
    for l in pipeline_model.layers:
        l.trainable = False
        if 'lst' in l.name:
            l.stateful = True

    logging.info("Model ready")

    stop = False
    while not stop:        
        item = fq.get()
        if item is None:   
            logging.debug("Took item from img_q: None")
            stop = True
        else:
            imgs = item
            logging.debug("Took item from img_q: batch %s" % (imgs.shape, ))

        clas = pipeline_model.predict(imgs)
        for i in range(len(imgs)):
            img, cla = imgs[i], clas[i]
            logging.debug("Putting classified image %s into pred_q" % str(cla.shape))
            vq.put((img, cla))

        fq.task_done()
    
    logging.debug("Done with feeding model")
    vq.put(None)

def feed_video():
    frames = 1
    last_time = time.time()

    while True:
        item = vq.get()
        if item is None:
            logging.debug("Last frame item received")
            vq.task_done()
            break
        
        x, yp = item        
        logging.debug("Received frame with shapes %s and %s" % (x.shape, yp.shape, ))

        x = np.squeeze(x)
        yp = np.squeeze(yp)
        yp = np.stack((yp[..., 3], yp[..., 2], yp[..., 1],), axis=-1) # bgr 

        # Input
        im_x.cla()
        im_x.set_axis_off()
        im_x.imshow(np.squeeze(x), cmap='gray') # last image from sequence
        im_x.set_title("Input")
        # Prediction
        im_y.cla()
        im_y.set_axis_off()
        im_y.imshow(yp)
        im_y.set_title("Predicted (next frame)")

        # Render into numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        im = np.reshape(im, (h, w, 3))

        if args.interactive:
            cv2.imshow("Output", im) 
        else:
            out.write(im) 
        logging.debug("Wrote frame with shape %s to video" % (im.shape, ))

        if frames % 100 == 0:
            logging.info("Frame %d (%.2f FPS)" % (frames, 100. / (time.time() - last_time)))       
            last_time = time.time()
        frames += 1

        vq.task_done()

    logging.debug("Done with writing video")

def preprocess(img, image_shape=(800, 800)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    img_h, img_w = img.shape[:2]

    # invert
    threshold = 100
    img_line = img.copy()
    img_line[img >= threshold] = 255
    img_line = (img_line < threshold).astype(np.uint8)

    # open and dilate 
    # (25, 9) (25, 7) kernel dimensions for (2448, 2048) images
    kernel_dims = (int(0.0156 * img_w), int(0.0045 * img_h))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dims)
    img_line = cv2.morphologyEx(img_line, cv2.MORPH_OPEN, kernel)
    kernel_dims = (int(0.0156 * img_w), int(0.0034 * img_h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
    img_line = cv2.dilate(img_line, kernel, iterations=1)

    # find line and undistort
    points = cv2.findNonZero((img_line > 0.5).astype(np.uint8))
    x = [p[0][0] for p in points]
    y = [p[0][1] for p in points]
    line = np.poly1d(np.polyfit(x, y, 2))

    xl = np.arange(0, img.shape[1])
    y = line(xl)
    y_min = np.min(y)
    y_shift = (y - y_min).astype(np.uint8)

    # cut into pieces and roll so the line ends up straigth
    parts = list()
    
    idx = 0
    idx_start = 0
    last_s = y_shift[0]
    for idx, s in enumerate(y_shift):
        # start new chunk whenever the amount of pixels to shift changes
        if idx < len(y_shift) or y_shift[idx] != last_s:            
            # get the chunk
            part = img[:, idx_start:idx]
            if last_s != 0:
                # roll upwards if there is something to roll
                part = np.roll(part, part.shape[0]-last_s, axis=0)
            
            parts.append(part)
            idx_start = idx
            
        last_s = s
    
    # stitch parts back together
    img = np.concatenate(parts, axis=1)    
    
    # create a new line
    newline = np.poly1d([y_min])
    y_cut = int(np.min(newline(xl)))

    img = img[:y_cut + 2, ...]

    d = img.shape[1] - image_shape[1]
    dl = int(d/2)
    dr = d-dl
    img = img[max(0, img.shape[0] - image_shape[0]):, dl:img.shape[1]-dr]

    if img.shape != (ih, iw, ic):
        img = cv2.resize(img, (iw, ih))
    return img

# Cannot be multithreaded because model is on GPU
t_feed = threading.Thread(target=feed_model)
t_feed.start()

# Cannot be multithreaded because matplotlib isnt threadsafe
t_video = threading.Thread(target=feed_video)
t_video.start()

# Start Threads
logging.debug("Filling queue with data")
status = True
i = 0

batch = []
while inp.grab():
    ret, frame = inp.retrieve()
    if not ret:
        raise RuntimeWarning("Error while reading video stream")

    if args.preprocess:
        frame = preprocess(frame)

    batch.append(frame[np.newaxis, np.newaxis, ..., np.newaxis])

    if len(batch) == args.batch_size:
        fq.put(np.concatenate(batch, axis=0))
        batch.clear()
    if i > args.max_frames:
        fq.put(np.concatenate(batch, axis=0))
        break
    i += 1 

fq.put(None)
logging.debug("Waiting for work to finish...")
fq.join()
vq.join()
logging.debug("Collected queues")

t_feed.join()
t_video.join()

logging.debug("Collected threads")
 
logging.debug("Done")

if args.interactive:
    cv2.destroyAllWindows()
else:
    out.release()
    logging.debug("Writing video")