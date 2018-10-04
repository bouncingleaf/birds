"""
Take normalized data and make some neural nets

"""
import os
import re

import tensorflow as tf
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# Constants
TRAIN_PATH = 'C:/datasets/CUB_200_2011/processed/data224/train/'
TEST_PATH = 'C:/datasets/CUB_200_2011/processed/data224/test/'
PIXELS = 224
BATCH_SIZE = 3
NUM_CLASSES = 200

# From https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/ 
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
 
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))
 
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,
               num_filters):
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
 
    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
 
    layer += biases
 
    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
 
    return layer

 
def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
 
    return layer

 
def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
 
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
 
    return layer

def main():
    """
    Main program for running from command line

    """

    # Read the data as Pandas data frames
    train_images = pd.read_csv(os.path.join(TRAIN_PATH, 'train_images.csv'))
    train_labels = pd.read_csv(os.path.join(TRAIN_PATH, 'train_labels.csv'))
    test_images = pd.read_csv(os.path.join(TEST_PATH, 'test_images.csv'))
    test_labels = pd.read_csv(os.path.join(TEST_PATH, 'test_labels.csv'))

    train_files = train_images.values.flatten()
    print(train_files)

    # Define placeholders
    # Code based on https://www.mabl.com/blog/image-classification-with-tensorflow
    # And https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
    x_images = tf.placeholder(tf.float32, shape=[None,PIXELS,PIXELS,3], name='x')
    y_labels = tf.placeholder(tf.int32, shape=[None,200], name='y')

    # Define variables
    weights = tf.Variable(tf.zeros[])
    biases = tf.Variable()

    # # train_files.map(lambda x: np.array(Image.open(os.path.join(TRAIN_PATH, x[0]))))
     
    # layer_conv1 = create_convolutional_layer(input=x,
    #             num_input_channels=num_channels,
    #             conv_filter_size=filter_size_conv1,
    #             num_filters=num_filters_conv1)

    # layer_conv2 = create_convolutional_layer(input=layer_conv1,
    #            num_input_channels=num_filters_conv1,
    #            conv_filter_size=filter_size_conv2,
    #            num_filters=num_filters_conv2)
 
    # layer_conv3= create_convolutional_layer(input=layer_conv2,
    #            num_input_channels=num_filters_conv2,
    #            conv_filter_size=filter_size_conv3,
    #            num_filters=num_filters_conv3)
          
    # layer_flat = create_flatten_layer(layer_conv3)
 
    # layer_fc1 = create_fc_layer(input=layer_flat,
    #                  num_inputs=layer_flat.get_shape()[1:4].num_elements(),
    #                  num_outputs=fc_layer_size,
    #                  use_relu=True)
 
    # layer_fc2 = create_fc_layer(input=layer_fc1,
    #                  num_inputs=fc_layer_size,
    #                  num_outputs=NUM_CLASSES,
    #                  use_relu=False)

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
    #                                                 labels=y_true)
    # cost = tf.reduce_mean(cross_entropy)

    # batch_size = 16
 
    # x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
 
    # feed_dict_train = {x: x_batch, y_true: y_true_batch}
 
    # session.run(optimizer, feed_dict=feed_dict_tr)

    # x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
 
    # feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}
 
    # val_loss = session.run(cost, feed_dict=feed_dict_val)

    # correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
    main()
