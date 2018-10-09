""" Build an Image Dataset in TensorFlow.
Based on code from:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

"""

from __future__ import print_function

import tensorflow as tf
import os
import argparse
import sys

FLAGS = None


def train():

    # Image Parameters
    N_CLASSES = 200     # Total number of classes
    CHANNELS = 3        # The 3 color channels,
    PIXELS = 224


    def read_images(dataset_path, batch_size):
        """
        Reading the dataset

        Based on code from:
        https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
        """
        imagepaths, labels = list(), list()

        # Read dataset file
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))

        # Convert to Tensor
        imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)


        def parse_images(filename, label):
            image = tf.read_file(filename)
            image = tf.image.decode_jpeg(image)
            image = tf.image.convert_image_dtype(image, tf.float32)
            return image

        dataset = tf.data.Dataset.from_tensor_slices((imagepaths, labels))
        dataset = dataset.map(parse_images).batch(FLAGS.batch_size)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        return next_element

    x_images = tf.placeholder(tf.float32, shape=[None,PIXELS,PIXELS,3], name='x')
    y_labels = tf.placeholder(tf.int32, shape=[None,200], name='y')

    # Define variables -- needs a better init
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



###############333

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


        # # Build a TF Queue, shuffle data
        # image, label = tf.train.slice_input_producer([imagepaths, labels],
        #                                              shuffle=True)


        # # Read images from disk
        # image = tf.read_file(image)
        # image = tf.image.decode_jpeg(image, channels=CHANNELS)
        # image = tf.image.convert_image_dtype(image, tf.float32)

        # # Create batches
        # X, Y = tf.train.batch([image, label], batch_size=batch_size,
        #                       capacity=batch_size * 8,
        #                       num_threads=4,
        #                       shapes=[[PIXELS,PIXELS,CHANNELS],[]])

        print("Done reading images in {}.".format(dataset_path))
        return X, Y

    # Create model
    def conv_net(x, n_classes, dropout, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):

            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # Convolution Layer with 32 filters and a kernel size of 5
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv2)

            # Fully connected layer (in contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)
            # Because 'softmax_cross_entropy_with_logits' already applies softmax,
            # we only apply softmax to testing network
            out = tf.nn.softmax(out) if not is_training else out

        return out

    TRAIN_PATH = os.path.join(FLAGS.file_dir, 'train/train_data.txt')
    TEST_PATH = os.path.join(FLAGS.file_dir, 'test/test_data.txt')
    x_train, y_train = read_images(TRAIN_PATH, FLAGS.batch_size)
    x_test, y_test = read_images(TEST_PATH, FLAGS.batch_size)

    # Because Dropout has different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that share the same weights.

    # Create a graph for training
    logits_train = conv_net(x_train, N_CLASSES, FLAGS.dropout, reuse=False, is_training=True)

    # Create another graph for testing that reuses the same weights
    logits_test = conv_net(x_test, N_CLASSES, FLAGS.dropout, reuse=True, is_training=False)

    # Define loss and optimizer (with train logits, for dropout to take effect)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=y_train))
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(y_test, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Saver object
    saver = tf.train.Saver()

    # Start training
    print("Training with batch size {}, learning rate {}, dropout {}, steps {}".format(
        FLAGS.batch_size, 
        FLAGS.learning_rate, 
        FLAGS.dropout,
        FLAGS.max_steps))

    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Start the data queue
        tf.train.start_queue_runners()

        # Training cycle
        for step in range(1, FLAGS.max_steps + 1):

            if step % FLAGS.display_steps == 0:
                # Run optimization and calculate batch loss and accuracy
                _, loss, acc = sess.run([train_op, loss_op, accuracy])
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
            else:
                # Only run the optimization op (backprop)
                sess.run(train_op)

        print("Optimization Finished!")
         # Save your model
        saver.save(sess, FLAGS.file_dir)

def main(_):
    train()

if __name__ == '__main__':
    # based on code from 
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--display_steps', type=int, default=50,
                        help='Print status at intervals of this many steps.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--batch_size', type=float, default=10,
                        help='Batch size.')
    parser.add_argument(
        '--file_dir',
        type=str,
        # default='/Users/leaf/CS767',
        default='C:/datasets/CUB_200_2011/processed/data224/',
        help='Base file directory')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/birds_with_summaries',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)