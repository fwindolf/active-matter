import argparse
from universal_datagen.generator.generator_text import AM2018TxtGenerator
from universal_models.models.models import get_model

import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)
import keras

import glob
import time
import queue
import threading
import logging

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %I:%M:%S', level=logging.INFO) # like tf

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('-m', '--model', help='Name of the model to train', required=True)

parser.add_argument('-d', '--data', help='Path to the data sequence', required=True)
parser.add_argument('-s', '--structure', help='Structure of the data', required=True, choices=['pair', 'stacked', 'sequence'])       
parser.add_argument('-w', '--weights_folder', help='Path to the stored weights of the model', required=True)
parser.add_argument('-f', '--fps', type=float, default=20.0, help='FPS for the video')
parser.add_argument('-o', '--output_name', default='output', help='Name of the video')
parser.add_argument('-c', '--classes', help='Number of classes in output data', type=int, default=4)
parser.add_argument('-b', '--batch_size', help='Speed up by increasing the number of frames fed to the model', type=int, default=1)
parser.add_argument('-max', '--max_frames', help='Maximum number of frames to process', type=int, default=100000)

parser.add_argument('-ih', '--input_height', help='Height dimension of input data to model', type=int, default=256)
parser.add_argument('-iw', '--input_width', help='Width dimension of input data to model', type=int, default=256)
parser.add_argument('-in', '--input_channels', help='Channels dimension of input data to model', type=int, default=1)

args = parser.parse_args()

data = AM2018TxtGenerator([args.data,], (args.input_height, args.input_width, args.input_channels), 
                                      (args.input_height, args.input_width, args.classes))

def fig_to_np(fig):
    # Render into numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    im = np.reshape(im, (h, w, 3)) # rgb
    im = np.stack((im[..., 2], im[..., 1], im[..., 0],), axis=2) # bgr

    return im


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im_x, im_y = axes.ravel()

x = np.zeros((args.input_height, args.input_width))
y = np.zeros((args.input_height, args.input_width))
im_x.set_axis_off()
im_x.imshow(x, cmap='hot')
im_x.set_title("Input")

im_y.set_axis_off()
im_y.imshow(y, cmap='hot')
im_y.set_title("Prediction")

im = fig_to_np(fig)
h, w, _ = im.shape

print("Creating video writer")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.output_name + '.mp4',fourcc, args.fps, (w, h)) # w, h

# Test video writer
try:
    out.write(im)
except cv2.error as e:
    logging.critical("Tested to write empty frame to video but failed. This might indicate wrong image/video dimensions")
    exit(0)

data_iterator = data.iterator(args.structure, labeled=True, cropped=False, ordering='channel_first')

fq = queue.Queue(30)
vq = queue.Queue(30)

def feed_model():
    # create model in this thread
    model, _, _ = get_model(args.model, args.input_height, args.input_width, args.input_channels, args.classes)
    weights = glob.glob(args.weights_folder + "/*.hdf5")
    model.load_weights(sorted(weights)[-1])

    logging.debug("Model created")
    stop = False    
    while not stop:
        xb = []
        while(len(xb) < args.batch_size):
            item = fq.get()
            if item is None:   
                logging.debug("Last data item received")         
                stop = True
                fq.task_done()
                break

            x, _ = item
            logging.debug("Received data item with shape %s" % (x.shape, ))
            xb.append(x)
            fq.task_done()  
        
        if len(xb) > 0:
            ypb = model.predict(np.asarray(xb))
        
            logging.debug("New batch of %d items ready" % len(xb))
            for i in range(len(xb)):
                vq.put((xb[i], ypb[i]))
    
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
        x = np.moveaxis(x, 0, -1) # make channel_last
        yp = np.reshape(yp[..., 1:], (args.input_height, args.input_width, 3)) # bgr
        yp = np.stack((yp[..., 2], yp[..., 1], yp[..., 0],), axis=2) # bgr 

        # Input
        im_x.cla()
        im_x.set_axis_off()
        im_x.imshow(np.squeeze(x[..., -1]), cmap='gray') # last image from sequence
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
        out.write(im) 
        logging.debug("Wrote frame with shape %s to video" % (im.shape, ))

        if frames % 100 == 0:
            logging.info("Frame %d (%.2f FPS)" % (frames, 100. / (time.time() - last_time)))       
            last_time = time.time()
        frames += 1

        vq.task_done()

    logging.debug("Done with writing video")

# Start Threads
logging.debug("Starting threads")
t_feed = threading.Thread(target=feed_model)
t_feed.start()
t_video = threading.Thread(target=feed_video)
t_video.start()

logging.debug("Filling queue with data")
# Fill queue with data
i = 0
for b in data_iterator:
    fq.put(b)

    if i > args.max_frames:
        logging.debug("Stopping with filling data after %d datapoints" % i)
        break
    i += 1

logging.debug("Inserting last data item")
fq.put(None)

logging.debug("Waiting for work to finish...")
fq.join()
vq.join()
logging.debug("Collected queues")

t_feed.join()
t_video.join()
logging.debug("Collected threads")
 
logging.debug("Done")
out.release()
logging.debug("Writing video")