import argparse
from universal_datagen.generator.generator_text import AM2018TxtGenerator
from universal_models.models.models import get_model

import tensorflow as tf
import keras
from keras.callbacks import *

from utils import *
import glob

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('-m', '--model', help='Name of the model to train',
                    required=True)
parser.add_argument('-s', '--structure', help='Structure of the data', 
                    choices=['pair', 'stacked']) # Not sequence (Time based networks...)
parser.add_argument('-l', '--labeled', help='Use labeled data',
                    action='store_true', default=False)                    

# dataset arguments
parser.add_argument('-dp', '--dataset_paths', nargs='+', help='Folders that contain data',
                    required=True)
parser.add_argument('-dh', '--dataset_input_height', help='Height dimension of input data to model',
                    type=int, default=256)
parser.add_argument('-dw', '--dataset_input_width', help='Width dimension of input data to model',
                    type=int, default=256)
parser.add_argument('-dn', '--dataset_input_channels', help='Channels dimension of input data to model',
                    type=int, default=1)
parser.add_argument('-dz', '--dataset_stack_size', help='Stack size of input data',
                    type=int, default=3)                    
parser.add_argument('-dc', '--dataset_num_classes', help='Number of classes for labels in dataset',
                    type=int, default=4)              
parser.add_argument('-ds', '--dataset_crop_scale', help='Prescaling before cropping',
                    type=float, default=None)    
parser.add_argument('-dm', '--dataset_max_num', help='Maximum number of data files to use to limit dataset size',
                    type=int, default=None)  


# training arguments
parser.add_argument('-to', '--train_optimizer', help='Optimizer used for training',
                    default='adam')
parser.add_argument('-tr', '--train_learning_rate', help='Learning rate for optimizer',
                    type=float, default=0.001)                    
parser.add_argument('-tm', '--train_metrics', help='Metrics to evaluate training progress',
                    type=list, default=['accuracy', ])
parser.add_argument('-tc', '--train_crops', help='Use crops of the original data',
                    type=int, default=None)
parser.add_argument('-te', '--train_epochs', help='Number of epochs the model is trained for',
                    type=int, default=30)                                                            
parser.add_argument('-tb', '--train_batchsize', help='Batchsize used for training',
                    type=int, default=8)                    
parser.add_argument('-ts', '--train_split', help='Split between the training and validation data',
                    type=float, default=0.2)                    

args = parser.parse_args()

input_height = args.dataset_input_height
input_width = args.dataset_input_width
input_channels = args.dataset_input_channels
n_classes = args.dataset_num_classes

# Create and Configure model
if args.labeled:
    model, output_height, output_width = get_model(args.model, input_height, input_width, input_channels * args.dataset_stack_size, n_classes)
    loss = 'categorical_crossentropy'
else:
    model, output_height, output_width = get_model(args.model, input_height, input_width, input_channels * args.dataset_stack_size, input_channels)
    loss = 'bce_dice_loss' # only works for 1D output

if args.train_optimizer == 'adam':
    optimizer = keras.optimizers.Adam(lr=args.train_learning_rate)
elif args.train_optimizer == 'adadelta':
    optimizer = keras.optimizers.Adadelta(lr=args.train_learning_rate)
else:
    raise AttributeError("Invalid optimizer")

model.compile(optimizer=optimizer, loss=loss, metrics=args.train_metrics)
model_output_dir = 'output/%s_ep%02d_nc%02d_st_%s_%s' % (args.model, args.train_epochs, n_classes, args.structure, "l" if args.labeled else "ul")
filepath = model_output_dir + '/weights_{epoch:03d}.hdf5'

print("Model output:", model_output_dir)

callbacks = []
callbacks.append(ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=10, min_lr=1e-5, factor=0.1))
callbacks.append(EarlyStopping(monitor='val_loss', patience=50))
callbacks.append(TensorBoard(log_dir = model_output_dir, histogram_freq=0))

# Configure Keras
config = tf.ConfigProto()
config.gpu_options.allow_growth=True # [pylint: ignore]
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# Create data generators
training_data = AM2018TxtGenerator(args.dataset_paths, (input_height, input_width, input_channels * args.dataset_stack_size), 
                                      (output_height, output_width, n_classes), crop_scale=args.dataset_crop_scale, 
                                      num_data=args.dataset_max_num)

tgen, vgen = training_data.generator(structure=args.structure, labeled=args.labeled, batch_size=args.train_batchsize, 
                                     num_crops=args.train_crops, split=args.train_split, flatten_label=True)

# Train model
print("Start Training on", len(training_data), "datapoints")
data = next(tgen)
print("Batches of ", data[0].shape, ",", data[1].shape)

t_steps = int((len(training_data) * (1 - args.train_split)) // args.train_batchsize)
v_steps = int((len(training_data) * args.train_split) // args.train_batchsize)

print("Training with", t_steps, "/", v_steps, "steps")
model.fit_generator(tgen,
                    epochs=args.train_epochs, 
                    steps_per_epoch=t_steps,
                    validation_data=vgen,
                    validation_steps=v_steps,                    
                    callbacks=callbacks,
                    shuffle=True, 
                    workers=2,
                    verbose=1)

print("Finish Training")

best_weights = glob.glob(model_output_dir + '/weights*')[-1]
model.load_weights(best_weights)

print("Evaluate on Validation data")
score = model.evaluate_generator(vgen, steps=1, verbose=1)

print("Evaluation: Accuracy %.2f%%" % (score[1] * 100))

# Visual eval on training data
x_train, y_train = next(tgen)
yp_train = model.predict(x_train)

# unflatten
yp_train = np.reshape(np.moveaxis(yp_train, -1, 1), (x_train.shape[0], 4, output_height, output_width))
y_train = np.reshape(np.moveaxis(y_train, -1, 1), (x_train.shape[0], 4, output_height, output_width))

n_c = n_classes if args.labeled else input_channels

data = (x_train, to_classes(yp_train, n_classes=n_c), y_train)
if args.structure == 'pair':
    fig = compare_pair(data, labeled=args.labeled)
elif args.structure == 'sequence':
    pass
    # fig = show_sequence(data)
elif args.structure == 'stacked':
    fig = compare_pair(data) # works by showing x as stacked

plt.savefig(model_output_dir + '/training.png')

# Visual eval on training data
x_val, y_val = next(vgen)
yp_val = model.predict(x_val)
yp_val = np.reshape(np.moveaxis(yp_val, -1, 1), (x_val.shape[0], 4, output_height, output_width))
y_val = np.reshape(np.moveaxis(y_val, -1, 1), (x_val.shape[0], 4, output_height, output_width))

data = (x_val, to_classes(yp_val, n_classes=n_c), y_val)
if args.structure == 'pair':
    fig = compare_pair(data, labeled=args.labeled)
elif args.structure == 'sequence':
    pass
    # fig = show_sequence(data)
elif args.structure == 'stacked':
    fig = compare_pair(data) # works by showing x as stacked

plt.savefig(model_output_dir + '/validation.png')