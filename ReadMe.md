# AM2018 Auto Labeling

This repository provides tools to automatically transfer classification from
simulation to experimental data.

## Active Matter - AM2018 Data

The AM2018 data exists of both experimental and simulation data, which can 
be used the same way via a common visualization in form of white circles on
black background.

There are 3 phases defined for the data, with background being phase 0:
- 1: solid
- 2: liquid
- 3: gas

### Experimental data

Experiments are done in a confined system with roller particles that are affected
both from gravity and from an electrical field, which propels them while they sediment 
towards the bottom of the system (if gravity is applied).
Experiments are take with various amount of gravity (angle of the system) and with 
various field strengths applied. 

This leads to different behaviours of the particle in the system - From sedimentation in a
solid phase towards the bottom of the system, to a gas-like state. Between the solid and 
the gas phase there is a (or even more) liquid-like phases in which particles are pulled
out of the solid phase or slow down into.

### Simulation data

The whole system is also simulated via an adaptation of the Viscek model to produce 
similar phenomenons as visible during the experiments. 

A benefit of the experimental data is the tracability of particle as well as the 
availabilty of a vast amount of metrics for particles(from velocity to forces acting 
on them). This makes it alot easier to classify the particles into the different phases.

### Inference

The final goal of this project is to being able to transfer knowledge about the phases in 
the system from the simulation to experimental data. Using `video_pipeline.py` with trained
models allows just that.

The output of the script are videos or a live video feed, while either videos or a camera
stream can be used. 

Videos look like this:


![Mainly Gas](gifs/0deg_-120V.gif)

![Mainly Liquid](gifs/60deg_-105V.gif)

![Mainly Solid](gifs/60deg_-95V.gif)

## Environment

### Installing on Triton

On trition, setting up the environment without care will exceed the quota in the home directory. Thus follow the [tutorial](http://scicomp.aalto.fi/triton/apps/python.html#conda) on howto use anaconda with triton.

To be able to use conda, load the suitable `anaconda3/<version>-gpu` module.
In order to train or evaluate, source the environment and load the modules `CUDA` and `cuDNN` (mind the big D)

To install the needed packages, proceed with the local introductions.

There is also a script to run training locally `run_local.sh` and on triton `run_train.sh`. 

### Installing locally

With anaconda3 installed (and loaded), the environment can be set up by executing:
`conda create --name <env-name> --file requirements.txt`

This will create a new conda environment and install all the necessary packages:
```
tensorflow-gpu # Tensorflow (DL) library, tested with version 1.6
keras-gpu      # Abstraction of Tensorflow
pillow         # Python Image Library, for maniuplating and loading images
matplotlib     # Plotting library
jupyter        # Interactive python in the browser
```

Some packages might not be available in your environment, this was tested on Ubuntu 16.10 and CentOS 7. The `conda-forge` channel might be worth checking out if that is the case.

For using the scripts to create videos, OpenCV is an additional requirement. Uncomment that part of the `requirements.txt` while creating the environment or manually run `conda install opencv`.

## Running 

Find out the usage options by running `train.py -h`. Help strings indicate the usage
of different options.

```
usage: train.py [-h] -m MODEL [-s {pair,stacked,sequence}] [-l]
                [-dt {text,image,mixed}] -dp DATASET_PATHS [DATASET_PATHS ...]
                [-dh DATASET_INPUT_HEIGHT] [-dw DATASET_INPUT_WIDTH]
                [-dn DATASET_INPUT_CHANNELS] [-dz DATASET_STACK_SIZE]
                [-dc DATASET_NUM_CLASSES] [-da DATASET_CROP_AREA]
                [-dm DATASET_MAX_NUM] [-to TRAIN_OPTIMIZER]
                [-tr TRAIN_LEARNING_RATE] [-tl {crossentropy,dice}]
                [-tc TRAIN_CROPS] [-te TRAIN_EPOCHS] [-tb TRAIN_BATCHSIZE]
                [-ts TRAIN_SPLIT]
```

`-m `: Name for the model (as string) you want to train. 

`-s `: Format of the data you want to use for training. `pair` data consists of a pair of a consecutive image and label, `stacked` data consists of `-dz` images stacked on top of each other and the label. `sequence` data consists of `-dz` pairs of images and labels.

`-l `: Flag to use labeled data. For labeled data activate, the data consists of images and labels. If omitted, images with a temporal offset of 1 are used as labels instead.

`-dt`: Decides which form of data should be used, creating the respective data generator for the form of data. Make sure the right type is set by checking the ouput during training, where it says how much datapoints the model is trained on.

`-dp`: Provide at least one path to the datasets, where each path should point to a location where the data files are located.

`-da`: The area of the image (ish) that should be used for cropping. For different sizes for image data, this makes sure that the area covered by crops is always the same. Needs `-tc` to be activated in order to have cropping.

`-dm`: The maximum number of data used for training. It will distribute the amount of data taken from different paths evenly. 

## Creating Video from models

Find out the full usage options by running `video.py -h`.

```
usage: video.py [-h] -m MODEL -d DATA -s {pair,stacked,sequence} -w
                WEIGHTS_FOLDER [-f FPS] [-o OUTPUT_NAME] [-c CLASSES]
                [-b BATCH_SIZE] [-max MAX_FRAMES] [-ih INPUT_HEIGHT]
                [-iw INPUT_WIDTH] [-in INPUT_CHANNELS]
```

Creating video only works (and produces meaningful results) when using
the same parameters as for training. 

`-w  `: The output folder generated during training.

`-max`: If there is more data in the `-d` directory than you want to use, simply 
specify the maximum number of frames for the video.
