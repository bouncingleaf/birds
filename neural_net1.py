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

# def get_data(data_in, path):
#     """
#     Get the image RGB data and reshape
#     """
#     rows = data_in.shape[0]
#     labels = np.empty(rows)
#     features = np.empty((rows,PIXELS,3,PIXELS))
#     i = 0
#     for item in data_in.itertuples():
#         labels[i] = item.class_id
#         image_path = os.path.join(path, item.path)
#         img = Image.open(image_path)
#         array = np.array(img)
#         array.reshape(3,PIXELS*PIXELS)
#         array = np.transpose(array,(0,2,1))
#         features[i] = array
#         i = i + 1
#     assert features.shape[0] == labels.shape[0]
#     return features, labels


def main():
    """
    Main program for running from command line

    """
    # bird_train = pd.read_csv(os.path.join(TRAIN_PATH, 'train_data.csv'))
    # bird_train_features, bird_train_labels = get_data(bird_train, TRAIN_PATH)
    # bird_train_data = tf.data.Dataset.from_tensor_slices((bird_train_features, bird_train_labels))
    # .batch(BATCH_SIZE)
    # train_iterator = bird_train_data.make_one_shot_iterator()
    # train_next_element = train_iterator.get_next()

    # bird_test = pd.read_csv(os.path.join(TEST_PATH, 'test_data.csv'))
    # bird_test_features, bird_test_labels = get_data(bird_test, TEST_PATH) 
    # bird_test_data = tf.data.Dataset.from_tensor_slices((bird_test_features, bird_test_labels))
    # .batch(BATCH_SIZE)
    # test_iterator = bird_test_data.make_one_shot_iterator()
    # test_next_element = test_iterator.get_next()

    # Create the model
    # x_r = tf.placeholder(tf.float32,shape=[None,PIXELS*PIXELS])
    # x_g = tf.placeholder(tf.float32,shape=[None,PIXELS*PIXELS])
    # x_b = tf.placeholder(tf.float32,shape=[None,PIXELS*PIXELS])

    # # 200 because 0-9 possible numbers
    # W_r = tf.Variable(tf.zeros([PIXELS*PIXELS,200]))
    # W_g = tf.Variable(tf.zeros([PIXELS*PIXELS,200]))
    # W_b = tf.Variable(tf.zeros([PIXELS*PIXELS,200]))

    # b = tf.Variable(tf.zeros([200]))

    # # Create the Graph
    # y = tf.matmul(x_r,W_r) + tf.matmul(x_g,W_g) + tf.matmul(x_b,W_b) + b 

    # # Loss and Optimizer

    # y_true = tf.placeholder(tf.float32,[None,200])

    # # Cross Entropy

    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))      

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

    # train = optimizer.minimize(cross_entropy)

    # ### Create Session

    # init = tf.global_variables_initializer()

    # with tf.Session() as sess:
    #     sess.run(init)
   
    #     # Train the model for 100 steps on the training set
    
    #     for step in range(10):
    #         batch_x, batch_y = sess.run(train_next_element)
    #         print(batch_x[0])
        
    #     sess.run(train,feed_dict={x_r:batch_x[0], x_g:batch_x[1], x_b:batch_x[2], y_true:batch_y})
        
    # # Test the Train Model
    # matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    
    # acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    
    # print(sess.run(acc,feed_dict={x:bird_test_features,y_true:bird_test_labels}))

if __name__ == '__main__':
    main()
