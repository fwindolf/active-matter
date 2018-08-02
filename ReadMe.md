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

## Environment

### Installing on Triton

On trition, setting up the environment will exceed the quota in the home directory. 
A few hacks later, you will be able to install and use conda packages from your work 
directory.

First, add the new environment and packages location to the `.condarc` file by executing:
`mkdir -p <path>/.conda/pkgs <path>/.conda/envs` 
`conda config --add pkgs_dirs <path>/.conda/pkgs`
`conda config --add envs_dirs <path>/.conda/envs`

Then, make sure the envs/pkgs directory exists in your home folder restrict the permissions
to those folders.
`mkdir -p /home/<user>/.conda/pkgs /home/<user>/.conda/envs`
`touch /home/<user>/.conda/environments.txt`
`chmod 444 /home/<user>/.conda/pkgs /home/<user>/.conda/envs`

Now proceed with the local introductions.

### Installing locally

With anaconda3 installed, the environment can be set up by executing:
`conda create --name <env-name> --file requirements.txt`

This will create a new conda environment and install all the necessary (and some more ;))
packages.

## Running 

Find out the usage options by running `train.py -h`. Help strings indicate the usage
of different options.

```
usage: train.py [-h] -m MODEL [-s {pair,stacked}] [-l] -dp DATASET_PATHS
                [DATASET_PATHS ...] [-dh DATASET_INPUT_HEIGHT]
                [-dw DATASET_INPUT_WIDTH] [-dn DATASET_INPUT_CHANNELS]
                [-dz DATASET_STACK_SIZE] [-dc DATASET_NUM_CLASSES]
                [-ds DATASET_CROP_SCALE] [-dm DATASET_MAX_NUM]
                [-to TRAIN_OPTIMIZER] [-tr TRAIN_LEARNING_RATE]
                [-tm TRAIN_METRICS] [-tc TRAIN_CROPS] [-te TRAIN_EPOCHS]
                [-tb TRAIN_BATCHSIZE] [-ts TRAIN_SPLIT]
```

`-m `: Name for the model (as string) you want to train. 

`-s `: Format of the data you want to use for training. `pair` data consists of
a pair of a consecutive image and label, `stacked` data consists of `-dz` images stacked on
top of each other and the label.

`-l `: Flag to use labeled data. When not using labeled data, the labels' image is used
instead.

`-dp`: Provide at least one path to the datasets, where each path should point
to a location where the data files are located.


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
