""" Build an Image Dataset in TensorFlow.
Based on code from:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

"""

import tensorflow as tf
import os
import argparse
import sys

FLAGS = None


def train():

    # Image Parameters
    N_CLASSES = 200     # Total number of classes
    CHANNELS = 3        # The 3 color channels,
    PIXELS = 128


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

        writer = tf.summary.FileWriter(os.path.join(FLAGS.file_dir, FLAGS.log_dir))
        writer.add_graph(sess.graph)

        # Run the initializer
        sess.run(init)

        # Start the data queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Training cycle
        print("Starting training")
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

        # Let python shut down cleanly
        coord.request_stop()
        coord.join(threads)

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
    parser.add_argument('--display_steps', type=int, default=100,
                        help='Print status at intervals of this many steps.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--batch_size', type=float, default=200,
                        help='Batch size.')
    parser.add_argument(
        '--file_dir',
        type=str,
        # default='/Users/leaf/CS767/data128/',
        default='C:/datasets/CUB_200_2011/processed/data128/',
        help='Base file directory')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/birds_with_summaries',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)