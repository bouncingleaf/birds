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
total_iterations = 0

def train():

    # Image Parameters
    CLASSES = 200     # Total number of classes
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

        # Build a TF Queue, shuffle data
        image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                     shuffle=True)

        # Read images from disk
        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=CHANNELS)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Create batches
        X, Y = tf.train.batch([image, label], batch_size=batch_size,
                              capacity=batch_size * 16,
                              num_threads=4,
                              shapes=[[PIXELS,PIXELS,CHANNELS],[]])

        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # print(sess.run(tf.shape(X)))

        print("Done reading images in {}.".format(dataset_path))
        return X, Y


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
        #We know that the shape of the layer will be [batch_size PIXELS PIXELS CHANNELS] 
        # But let's get it from the previous layer.
        layer_shape = layer.get_shape()
    
        ## Number of features will be img_height * img_width* CHANNELS. But we shall calculate it in place of hard-coding it.
        num_features = layer_shape[1:4].num_elements()
    
        ## Now, we Flatten the layer so we shall have to reshape to num_features
        layer = tf.reshape(layer, [-1, num_features])

        return layer
    
    
    def create_fc_layer(input,          
                 num_inputs,    
                 num_outputs,
                 use_relu=True):
    
        #Let's define trainable weights and biases.
        weights = create_weights(shape=[num_inputs, num_outputs])
        biases = create_biases(num_outputs)
    
        # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer


    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    # data = dataset.read_train_sets(train_path, PIXELS, classes, validation_size=validation_size)

    TRAIN_PATH = os.path.join(FLAGS.file_dir, 'train/train_data.txt')
    TEST_PATH = os.path.join(FLAGS.file_dir, 'test/test_data.txt')
    x_train, y_train = read_images(TRAIN_PATH, FLAGS.batch_size)
    x_test, y_test = read_images(TEST_PATH, FLAGS.batch_size)

    x = tf.placeholder(tf.float32, shape=[None, PIXELS,PIXELS,CHANNELS], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, CLASSES], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1, name="y_true_cls")
    
    ##Network graph params
    filter_size_conv1 = 3 
    num_filters_conv1 = 32

    filter_size_conv2 = 3
    num_filters_conv2 = 32

    filter_size_conv3 = 3
    num_filters_conv3 = 64
    
    fc_layer_size = 128
    
    layer_conv1 = create_convolutional_layer(input=x,
                   num_input_channels=CHANNELS,
                   conv_filter_size=filter_size_conv1,
                   num_filters=num_filters_conv1)
    layer_conv2 = create_convolutional_layer(input=layer_conv1,
                   num_input_channels=num_filters_conv1,
                   conv_filter_size=filter_size_conv2,
                   num_filters=num_filters_conv2)
    
    layer_conv3= create_convolutional_layer(input=layer_conv2,
                   num_input_channels=num_filters_conv2,
                   conv_filter_size=filter_size_conv3,
                   num_filters=num_filters_conv3)
              
    layer_flat = create_flatten_layer(layer_conv3)
    
    layer_fc1 = create_fc_layer(input=layer_flat,
                         num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                         num_outputs=fc_layer_size,
                         use_relu=True)
    
    layer_fc2 = create_fc_layer(input=layer_fc1,
                         num_inputs=fc_layer_size,
                         num_outputs=CLASSES,
                         use_relu=False) 
    
    y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
    y_pred_cls = tf.argmax(y_pred, dimension=1, name='y_pred_cls')


    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())    
    
    def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
        acc = sess.run(accuracy, feed_dict=feed_dict_train)
        val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))
        
    saver = tf.train.Saver()

    def train_the_model(num_iteration):
        global total_iterations
        
        for i in range(total_iterations,
                       total_iterations + num_iteration):
    
            # x_batch, y_true_batch, _, cls_batch = data.train.next_batch(FLAGS.batch_size)
            # x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(FLAGS.batch_size)
    
            
            feed_dict_tr = {x: x_train,
                               y_true: y_train}
            feed_dict_val = {x: x_test,
                                  y_true: y_test}
    
            sess.run(optimizer, feed_dict=feed_dict_tr)
    
            if i % FLAGS.display_steps == 0: 
                val_loss = sess.run(cost, feed_dict=feed_dict_val)
                epoch = int(i / FLAGS.display_steps)    
                
                show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                saver.save(sess, 'birds-model') 
        
    
        total_iterations += num_iteration
    
    train_the_model(num_iteration=FLAGS.max_steps)
     
        # # Create model
        # def conv_net(x, n_classes, dropout, reuse, is_training):
        #     # Define a scope for reusing the variables
    #     with tf.variable_scope('ConvNet', reuse=reuse):

    #         # Convolution Layer with 32 filters and a kernel size of 5
    #         conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    #         # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    #         conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

    #         # Convolution Layer with 32 filters and a kernel size of 5
    #         conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    #         # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    #         conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

    #         # Flatten the data to a 1-D vector for the fully connected layer
    #         fc1 = tf.contrib.layers.flatten(conv2)

    #         # Fully connected layer (in contrib folder for now)
    #         fc1 = tf.layers.dense(fc1, 1024)
    #         # Apply Dropout (if is_training is False, dropout is not applied)
    #         fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

    #         # Output layer, class prediction
    #         out = tf.layers.dense(fc1, n_classes)
    #         # Because 'softmax_cross_entropy_with_logits' already applies softmax,
    #         # we only apply softmax to testing network
    #         out = tf.nn.softmax(out) if not is_training else out

    #     return out

    # TRAIN_PATH = os.path.join(FLAGS.file_dir, 'train/train_data.txt')
    # TEST_PATH = os.path.join(FLAGS.file_dir, 'test/test_data.txt')
    # x_train, y_train = read_images(TRAIN_PATH, FLAGS.batch_size)
    # x_test, y_test = read_images(TEST_PATH, FLAGS.batch_size)

    # # Because Dropout has different behavior at training and prediction time, we
    # # need to create 2 distinct computation graphs that share the same weights.

    # # Create a graph for training
    # logits_train = conv_net(x_train, CLASSES, FLAGS.dropout, reuse=False, is_training=True)

    # # Create another graph for testing that reuses the same weights
    # logits_test = conv_net(x_test, CLASSES, FLAGS.dropout, reuse=True, is_training=False)

    # # Define loss and optimizer (with train logits, for dropout to take effect)
    # loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=logits_train, labels=y_train))
    # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    # train_op = optimizer.minimize(loss_op)

    # # Evaluate model (with test logits, for dropout to be disabled)
    # correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(y_test, tf.int64))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # # Initialize the variables (i.e. assign their default value)
    # init = tf.global_variables_initializer()

    # # Saver object
    # saver = tf.train.Saver()

    # # Start training
    # print("Training with batch size {}, learning rate {}, dropout {}, steps {}".format(
    #     FLAGS.batch_size, 
    #     FLAGS.learning_rate, 
    #     FLAGS.dropout,
    #     FLAGS.max_steps))

    # with tf.Session() as sess:

    #     writer = tf.summary.FileWriter(os.path.join(FLAGS.file_dir, FLAGS.log_dir))
    #     writer.add_graph(sess.graph)

    #     # Run the initializer
    #     sess.run(init)


    #     coord = tf.train.Coordinator()

    #     # Start the data queue
    #     threads = tf.train.start_queue_runners(coord=coord)

    #     # Training cycle
    #     for step in range(1, FLAGS.max_steps + 1):

    #         if step % FLAGS.display_steps == 0:
    #             # Run optimization and calculate batch loss and accuracy
    #             _, loss, acc = sess.run([train_op, loss_op, accuracy])
    #             print("Step " + str(step) + ", Minibatch Loss= " + \
    #                   "{:.4f}".format(loss) + ", Training Accuracy= " + \
    #                   "{:.3f}".format(acc))
    #         else:
    #             # Only run the optimization op (backprop)
    #             sess.run(train_op)

    #     coord.request_stop()
    #     coord.join(threads)

    #     print("Optimization Finished!")
    #      # Save your model
    #     saver.save(sess, FLAGS.file_dir)

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