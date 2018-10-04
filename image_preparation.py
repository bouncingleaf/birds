"""
Based on https://www.kaggle.com/gauss256/preprocess-images

By default this script will normalize the image luminance and resize to a
square image of side length 224. The resizing preserves the aspect ratio and
adds gray bars as necessary to make them square. The resulting images are
stored in a folder named data224.

The location of the input and output files, and the size of the images, can be
controlled through the processing parameters below.
"""
from multiprocessing import Process
import os
import re

import numpy as np
import pandas as pd
import PIL
from PIL import Image


# Processing parameters
SIZE = 224      # for ImageNet models compatibility
DATA_PATH="C:/datasets/CUB_200_2011/CUB_200_2011/"
OUTPUT_PATH = 'C:/datasets/CUB_200_2011/processed/'


def natural_key(string_):
    """
    Define sort key that is integer-aware

    This code is directly from https://www.kaggle.com/gauss256/preprocess-images
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] conbirdst stretch
    This code is directly from https://www.kaggle.com/gauss256/preprocess-images
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    # img_y_np = np.asarray(img_y).astype(float)
    img_y_np = np.asarray(img_y, dtype=np.float32)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm

def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    This code is directly from https://www.kaggle.com/gauss256/preprocess-images
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new('RGB', (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


def prep_images(image_data, out_dir):
    """
    Preprocess images

    Reads images in paths, and writes to out_dir
    This code is modified from https://www.kaggle.com/gauss256/preprocess-images
    """
    for item in image_data.itertuples():
        path = os.path.join(DATA_PATH, 'images', item.path)
        img = Image.open(path)
        img = img.crop((item.left, item.upper, item.right, item.lower))
        img = resize_image(norm_image(img), SIZE)
        bird_path = os.path.join(out_dir,os.path.dirname(item.path))
        os.makedirs(bird_path, exist_ok=True)
        full_image_path = os.path.join(bird_path, os.path.basename(item.path))
        img.save(full_image_path)


def load_images():
    """
    Load the image data from various files
    This code is my own
    """
    # Goldfinches are 2670-2709
    # START = 0      # inclusive
    # END = 100      # inclusive
    # Read in the bounding boxes and image_ids
    boxes = pd.read_csv(DATA_PATH + "bounding_boxes.txt",' ',names=['image_id','left','upper','width','height'])
    # boxes = boxes[boxes['image_id'].between(START,END)]
    # Set up right and lower values, drop the width and height
    boxes['right'] = boxes['left'] + boxes['width']
    boxes['lower'] = boxes['upper'] + boxes['height']
    boxes.drop(['width','height'], axis=1, inplace=True)
    # Read in the class names, connect each to the image id
    classes = pd.read_csv(DATA_PATH + "image_class_labels.txt",' ',names=['image_id','class_id'])
    classes['class_id'] = classes['class_id'] - 1
    images = pd.merge(boxes,classes,on='image_id')
    # Read in the names of the image files, connect each to its image_id
    files = pd.read_csv(DATA_PATH + "images.txt",' ',names=['image_id','path'])
    # files = files[files['image_id'].between(START,END)]
    images = pd.merge(images,files,on='image_id')
    # Merge with the train-test split indicators
    split = pd.read_csv(DATA_PATH + "train_test_split.txt",' ',names=['image_id','is_train'])
    images = pd.merge(images,split,on='image_id')
    # Actually split the data
    train = images[images['is_train'] == 1]
    test = images[images['is_train'] == 0]
    # Tried to use inplace here, but because I'm modifying I want copies of the data not the original
    train = train.drop('is_train',axis=1).set_index('image_id')
    test = test.drop('is_train',axis=1).set_index('image_id')
    train['path'] = train['path'].map(lambda x: 'C:/datasets/CUB_200_2011/processed/data224/train/' + x)
    test['path'] = test['path'].map(lambda x: 'C:/datasets/CUB_200_2011/processed/data224/test/' + x)
    return train, test

def main():
    """
    Main program for running from command line

    This code is modified from https://www.kaggle.com/gauss256/preprocess-images
    """
    # load Pandas dataframes with all the image data
    train, test = load_images()
    print('Images loaded')

    # Make the output directories
    base_out = os.path.join(OUTPUT_PATH, 'data{}'.format(SIZE))
    train_dir_out = os.path.join(base_out, 'train')
    test_dir_out = os.path.join(base_out, 'test')
    os.makedirs(train_dir_out, exist_ok=True)
    os.makedirs(test_dir_out, exist_ok=True)

    # Write the data to  files within each output directory
    train.to_csv(os.path.join(train_dir_out, 'train_data.txt'), sep=' ', columns=['path','class_id'], header=False, index=False)
    test.to_csv(os.path.join(test_dir_out, 'test_data.txt'), sep=' ', columns=['path','class_id'], header=False, index=False)

    # Preprocess the training files
    # procs = dict()
    # procs[1] = Process(target=prep_images, args=(train, train_dir_out, ))
    # procs[1].start()
    # procs[2] = Process(target=prep_images, args=(test, test_dir_out, ))
    # procs[2].start()

    # procs[1].join()
    # procs[2].join()


if __name__ == '__main__':
    main()
