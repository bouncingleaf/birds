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

