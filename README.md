# Jessica Roy's Final Project for MET CS 767

## About

This is Jessica Roy's Final Project for Boston University MET CS 602 online, Professor Eric Braude, fall term 1 of 2018. It is a convolutional neural network to classify bird images from the CUB-200-2011 dataset by species.

## Contents

- nn.py - The neural net
- image_preparation.py - The program for preparing the images. It takes a few minutes to run, and it only needs to be run once.

## Running

### To prepare the neural net to run:

1. Get the CUB-200-2011 dataset.
2. Edit image_preparation.py to reflect where it is installed
3. Run image_preparation.py. This will normalize the image files and crop them to the bounding boxes.
   It also produces text files the neural net training program can use.

You can comment out the "procs" section  near the end of image_preparation.py in order to rebuild the text file without re-cropping the images. Much faster!

## Results

### Round 1
Training accuracy remained 0.00 no matter how many epochs, and the test part of the algorithm crashed. Hmm.

### Round 2
Increased batch size (I think it was from 3 to 100). Training accuracy began to improve! But it got suspiciously high. 100% after five epochs. And the test part of the algorithm still crashed. 

### Round 3
Fixed the crash in the test algorithm, and testing accuracy was around 5%. Sounds like overfitting to me.

### Round 4
Flipped the images and rotated them slightly and cropped them slightly differently to make twice as many training examples. Training accuracy still suspiciously high:
    Epoch [6/6], Step [120/120], Loss: 0.5243, Accuracy: 87.50%
The test part of my algorithm crashed again... on a print statement, no less.

