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

parser.add_argument('-l', '--log_level', help='Set the level for logging', choices=['debug', 'info', 'warn', 'error'], default='info')

args = parser.parse_args()

logging.basicConfig(format='%(thread)d | %(levelname)s | %(asctime)s.%(msecs)03d: %(message)s', level=logging.getLevelName(args.log_level.upper())) # format like tf

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
    im = np.stack([im[..., 2], im[..., 1], im[..., 0]], axis=-1) # bgr
    return im

# Initialize image shape
fig, axes = plt.subplots(1, 2, figsize=(int(2.1 * lw/64), int(lh/64)))
plt.tight_layout()
im_x, im_y = axes.ravel()

x = np.zeros((ih, iw))
y = np.zeros((lh, lw))

im_x.set_axis_off()
im_x.imshow(x, cmap='hot')
im_x.set_title("Input")

im_y.set_axis_off()
im_y.imshow(y, cmap='hot')
im_y.set_title("Prediction")

im = fig_to_np(fig)
h, w, _ = im.shape

if args.output_name is None and args.interactive is False:
    logging.error("No output name found and not interactive, please provide one of both")
    exit(0)

if args.interactive:
    logging.info("Creating canvas")
    plt.ion()
else:
    logging.info("Creating video writer")
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

if args.interactive:
    qlen = 1
else:
    qlen = 30

fq = queue.Queue(qlen)
vq = queue.Queue(qlen)

cv_model = threading.Condition()
cv_draw = threading.Condition()
cv_drawn = threading.Condition()

def feed_images():
    logging.info("Creating video capture")
    cap = cv2.VideoCapture()
    status = cap.open(args.data)
    if not status:
        logging.error("Unable to open video (stream) %s" % args.data)
        os._exit(-1)

    # Get the fps to delay feeding images to actual speed
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # Wait for model to be ready
    with cv_model:
        cv_model.wait()

    logging.debug("Filling queue with data")

    batch = []

    frames = 0
    dropped_frames = 0
    while cap.grab():
        t_frame = time.time()
        ret, frame = cap.retrieve()
        if not ret:
            raise RuntimeWarning("Error while reading video stream")

        if args.preprocess:
            frame = preprocess(frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            fh, fw = frame.shape[:2]
            fch, fcw = int(fh/2), int(fw/2)

            # crop if the aspect ratio is different
            if (fh/fw) != (ih/iw):
                iratio = ih/iw
                if fh > fw:
                    fh = iratio * fw
                else:
                    fw = fh/iratio

                foh, fow = int(fh/2), int(fw/2)
                frame = frame[fch-foh:fch+foh, fcw-fow:fcw+fow]

            frame = cv2.resize(frame, (iw, ih))

        batch.append(frame[np.newaxis, np.newaxis, ..., np.newaxis])
        if args.interactive:
            logging.debug("Putting image to queue")
            try:
                fq.put_nowait(np.concatenate(batch, axis=0))
            except queue.Full:
                logging.debug("Dropping frame because queue is full!")
                dropped_frames += 1
            
            # Sleep to get to actual 
            wait = (1./cap_fps) - (time.time() - t_frame)
            time.sleep(max(0, wait))
            t_frame = time.time()
            batch.clear()
        elif len(batch) == args.batch_size:
            logging.debug("Putting batch to queue")
            fq.put(np.concatenate(batch, axis=0))
            logging.debug("fq now has %d items" % vq.qsize())
            batch.clear()
        if frames > args.max_frames:
            if len(batch) > 0:
                logging.debug("Putting last batch to queue")                        
                fq.put(np.concatenate(batch, axis=0))
                logging.debug("fq now has %d items" % vq.qsize())
            break
        
        frames += 1

    logging.debug("Image acquisition thread finished")
    logging.info("Processed %d frames, of which %d were dropped (%.1f%%)" % (frames, dropped_frames, 100*dropped_frames/frames))

    fq.put(None)
    logging.debug("Waiting for work to finish...")

def feed_model():
    # create model in this thread
    logging.debug("Loading model file %s" % (args.abstraction_model_dir + 'model.json'))
    with open(args.abstraction_model_dir + '/model.json', 'r') as f:
        abstract_model = model_from_json(f.read())

    weights = glob.glob(args.abstraction_model_dir + "weights*.hdf5")
    logging.debug("Loading weights file %s" % (sorted(weights)[-1]))
    abstract_model.load_weights(sorted(weights)[-1])

    # To be able to feed single images to the network, set the number of timesteps to 1
    # and exchange the input layers
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

    # To be able to feed single images to the network, set the number of timesteps to 1
    # and exchange the input layers                 
    new_input = Input(shape=(1, lh, lw, lc)) # time step of 1
    classify_model.layers.pop(0)
    new_output = classify_model(new_input)

    classify_model = Model(new_input, new_output)

    logging.debug("Classification model created")

    # Chain the two models together, first abstraction then classification
    pipeline_model = Model(abstract_model.inputs, classify_model(abstract_model(abstract_model.inputs)))
    
    # Set the network to keep its state after predictions
    for l in pipeline_model.layers:
        l.trainable = False
        if 'lst' in l.name:
            l.stateful = True

    logging.info("Model ready")

    # Broadcast that the model is ready
    with cv_model:
        cv_model.notify_all()

    stop = False
    while not stop:        
        item = fq.get()
        if item is None:   
            logging.debug("Took item from img_q: None")
            stop = True
        else:
            imgs = item
            logging.debug("Took item from img_q: batch %s" % (imgs.shape, ))
        
        t = time.time()
        clas = pipeline_model.predict(imgs)
        logging.debug("TIME: Predicting classes | %d ms" % ((time.time() - t) * 1000))
        t = time.time()
        for i in range(len(imgs)):
            img, cla = imgs[i], clas[i]
            logging.debug("Putting classified image %s into pred_q" % str(cla.shape))
            vq.put((img, cla))
            logging.debug("pred_q now has %d items" % vq.qsize())
        logging.debug("TIME: Classes to queue  | %d ms" % ((time.time() - t) * 1000))
        
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
        
        t = time.time()
        x, yp = item        
        logging.debug("Received frame with shapes %s and %s" % (x.shape, yp.shape, ))
        
        x = np.squeeze(x)
        yp = np.squeeze(yp)[..., 1:]
        #yp = np.stack((yp[..., 3], yp[..., 2], yp[..., 1],), axis=-1) # bgr 

        logging.debug("TIME: Modifying data     | %d ms" % ((time.time() - t) * 1000))
        t = time.time()

        # Wait until its drawn before changing the plot
        logging.debug("Waiting until last image was drawn")
        if args.interactive:
            with cv_drawn:
                cv_drawn.wait(timeout=1)

        # Input
        im_x.cla()
        im_x.set_axis_off()
        im_x.imshow(np.squeeze(x), cmap='gray') # last image from sequence
        im_x.set_title("Input")
        # Prediction
        im_y.cla()
        im_y.set_axis_off()
        im_y.imshow(yp)
        im_y.set_title("Prediction")

        logging.debug("TIME: Painting data      | %d ms" % ((time.time() - t) * 1000))
        t = time.time()

        if args.interactive:
            # Signal that a new frame is ready to be drawn
            logging.debug("Notifying about new image to draw")
            with cv_draw:
                cv_draw.notify_all()
        else:
            # Render into numpy array
            im = fig_to_np(fig)
            logging.debug("TIME: Rendering figure   | %d ms" % ((time.time() - t) * 1000))
            t = time.time()
            out.write(im) 
            logging.debug("Wrote frame with shape %s to video" % (im.shape, ))

        if frames % 100 == 0:
            logging.info("Frame %d (%.2f FPS)" % (frames, 100. / (time.time() - last_time)))       
            last_time = time.time()
        frames += 1

        logging.debug("TIME: Output figure      | %d ms" % ((time.time() - t) * 1000))
        t = time.time()

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

t_images = threading.Thread(target=feed_images)
t_images.start()

if args.interactive:
    # Draw as long as the video thread produces images
    while t_video.is_alive():
        # Drawing only works in main thread
        logging.debug("Waiting for new image")
        with cv_draw:
            cv_draw.wait()
        
        logging.debug("Drawing new frame")    
        
        fig.show()
        fig.canvas.flush_events()

        logging.debug("Notifying about drawn image")
        with cv_drawn:
            cv_drawn.notify_all()

t_images.join()
t_feed.join()
t_video.join()
logging.debug("Collected threads")

fq.join()
vq.join()
logging.debug("Collected queues")
 
logging.debug("Done")

if not args.interactive:
    out.release()
    logging.debug("Writing video")