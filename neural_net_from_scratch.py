# from __future__ import print_function

import tensorflow as tf
import os
import argparse
import sys

FLAGS = None

# Image Parameters
CLASSES = 200       # Total number of classes
CHANNELS = 3        # The 3 color channels
IMG_SIZE = 128


def main(_):

    def read_images(dataset_path, batch_size):
        """
        Reading the dataset

        Based on code from:
        https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
        https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
        """
        imagepaths, labels = list(), list()

        # Read dataset file
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))

        # Convert to Tensor
        imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string, name="imagepaths")
        labels = tf.convert_to_tensor(labels, dtype=tf.int32, name="labels")

        # Build a TF Queue, shuffle data
        image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                     shuffle=True)

        label = tf.one_hot(label, CLASSES, name="one_hot_label")

        # Read images from disk
        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=CHANNELS)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Create batches
        X, Y = tf.train.batch([image, label], batch_size=batch_size,
                              capacity=batch_size * 16,
                              num_threads=4,
                              shapes=[[IMG_SIZE,IMG_SIZE,CHANNELS],[CLASSES]])

        print("Done reading images in {}.".format(dataset_path))
        return X, Y


        # dataset = tf.data.Dataset.from_tensor_slices((imagepaths, labels))
        # dataset = dataset.map(parse_data)

        # # Shuffle the ENTIRE data set, otherwise each batch may be all the same bird!
        # dataset = dataset.shuffle(buffer_size=6000)

        # # Now put in real batches
        # dataset = dataset.batch(batch_size)

        # iterator = dataset.make_one_shot_iterator()

        # return iterator.get_next()

        # # Convert labels to one hot encoding
        # one_hot = tf.one_hot(label, CLASSES, name="one_hot_label")

        # return image, one_hot


    """
    The following code is from, or based on:
    https://github.com/sankit1/cv-tricks.com/blob/master/Tensorflow-tutorials/tutorial-2-image-classifier/train.py

    """

    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    # data = dataset.read_train_sets(train_path, IMG_SIZE, CLASSES, validation_size=validation_size)
    TRAIN_PATH = os.path.join(FLAGS.file_dir, 'train/train_data.txt')
    TEST_PATH = os.path.join(FLAGS.file_dir, 'test/test_data.txt')
    train_features, train_labels = read_images(TRAIN_PATH, FLAGS.batch_size)
    test_features, test_labels = read_images(TEST_PATH, FLAGS.batch_size)

    print("Data has been loaded: ", train_features, train_labels, test_features, test_labels)

    # x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE,IMG_SIZE,CHANNELS], name='x')

    ## labels
    # y_true = tf.placeholder(tf.float32, shape=[None, CLASSES], name='y_true')
    # y_true_cls = tf.argmax(y_true, axis=1, name="y_true_cls")

    ##Network graph params
    filter_size_conv1 = 3 
    num_filters_conv1 = 32

    filter_size_conv2 = 3
    num_filters_conv2 = 32

    filter_size_conv3 = 3
    num_filters_conv3 = 64

    fc_layer_size = 128

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
        #We know that the shape of the layer will be [FLAGS.batch_size IMG_SIZE IMG_SIZE CHANNELS] 
        # But let's get it from the previous layer.
        layer_shape = layer.get_shape()

        ## Number of features will be img_height * img_width * CHANNELS. But we shall calculate it in place of hard-coding it.
        num_features = layer_shape[1:4].num_elements()

        ## Now, we Flatten the layer so we shall have to reshape to num_features
        layer = tf.reshape(layer, [-1, num_features])

        return layer


    def create_fc_layer(input,          
           num_inputs,    
           num_outputs,
           use_relu=True):

        # Define trainable weights and biases.
        weights = create_weights(shape=[num_inputs, num_outputs])
        biases = create_biases(num_outputs)

        # Fully connected layer takes input x and produces wx+b.  Since, these are matrices, we use matmul function in Tensorflow
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    layer_conv1 = create_convolutional_layer(input=train_features,
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

    y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

    y_pred_cls = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
        labels=y_true)
    cost = tf.reduce_mean(cross_entropy, name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    session = tf.Session()

    session.run(tf.global_variables_initializer()) 

    saver = tf.train.Saver()

    def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))

    def train(num_iteration):

        for i in range(0,num_iteration):

            # x_batch, y_true_batch, _, cls_batch = data.train.next_batch(FLAGS.batch_size)
            # x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(FLAGS.batch_size)

            
            feed_dict_train = {x: train_features,
            y_true: train_labels}
            feed_dict_test = {x: test_features,
            y_true: test_labels}

            session.run(optimizer, feed_dict=feed_dict_train)

            if i % int(data.train.num_examples/FLAGS.batch_size) == 0: 
                val_loss = session.run(cost, feed_dict=feed_dict_val)
                epoch = int(i / int(data.train.num_examples/FLAGS.batch_size))    
                
                show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                saver.save(session, 'dogs-cats-model') 


    train(num_iteration=3000)

    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             elem = sess.run(train_features)
    #             print(elem)
    #         except tf.errors.OutOfRangeError:
    #             print("End of training set")
    #             break

    #     while True:
    #         try:
    #             elem = sess.run(test_labels)
    #             print(elem)
    #         except tf.errors.OutOfRangeError:
    #             print("End of test set")
    #             break

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
    parser.add_argument('--batch_size', type=float, default=100,
        help='Batch size.')
    parser.add_argument(
        '--file_dir',
        type=str,
        # default='/Users/leaf/CS767',
        default='C:/datasets/CUB_200_2011/processed/data128/',
        help='Base file directory')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/birds_with_summaries',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)