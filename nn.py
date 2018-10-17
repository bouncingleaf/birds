import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.datasets
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import sys

FLAGS = None

class BirdDataset(Dataset):
    def __init__(self, image_path, csv_file, transform=None):

        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_path = image_path
        self.birds = pd.read_csv(csv_file, sep=' ', names=['path','class_id'])
        self.transform = transform

    def __len__(self):
        return len(self.birds)

    def __getitem__(self, idx):
        img_name = self.birds.iloc[idx, 0]
        img = Image.open(os.path.join(self.image_path, img_name))
        img = np.array(img)
        label = self.birds.iloc[idx, 1]
        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'label': torch.tensor(int(label)).type(torch.FloatTensor)}

class CNN(nn.Module):
    """ based on https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset-e589682df0c5 """
    def __init__(self, image_size, num_classes):
        self.num_classes = num_classes
        self.hidden_layer = 400
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(512*256*256, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # Apply softmax here?
        return out

def build_datasets(path, batch_size):
    train_csv = os.path.join(path, 'train/train_data.txt')
    test_csv = os.path.join(path, 'test/test_data.txt')
    train_dataset = BirdDataset(path, csv_file=train_csv, transform=ToTensor())
    test_dataset = BirdDataset(path, csv_file=test_csv, transform=ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Train the model
def train(model, train_loader, num_epochs, learning_rate, display_every):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    loss_list = []
    acc_list = []

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            # Run the forward pass
            outputs = model(data['image'])
            labels = data['label'].type(torch.LongTensor)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % display_every == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
    return loss_list, acc_list

# Test the model
def test(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            outputs = model(data['image'])
            _, predicted = torch.max(outputs.data, 1)
            labels = data['label'].type(torch.LongTensor)
            total += labels.size(0)
            print(predicted, labels)
            correct += (predicted == labels).sum().item()

        print('Accuracy on the test images: {} %'.format((correct / total) * 100))


def plot(loss_list, acc_list):
    p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='ConvNet results')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), 
           np.array(acc_list) * 100,
           y_range_name='Accuracy',
           color='red')
    show(p)

def get_dirs(image, output):
    # These are my directory names on my testing system (a Mac)
    # The defaults in FLAGS are my directory names on my coding system (my Windows laptop)
    # You can override them both with --image_dir and --output_dir
    IMAGE_BASE = '/Users/leaf/CS767/data128/' 
    OUTPUT_BASE = '/Users/leaf/CS767/birds/output'

    if os.path.exists('/Users/leaf/CS767/'):
        image_dir = IMAGE_BASE
        output_dir = OUTPUT_BASE 
    else:
        image_dir = image
        output_dir = output
    if not os.path.exists(image_dir):
        print("Not a valid image directory {}, try using --image_dir flag".format(image_dir))
        return None, None
    if not os.path.exists(output_dir):
        print("Not a valid output directory {}, try using --output_dir flag".format(output_dir))
        return None, None
    else:
        return image_dir, output_dir

def main():
    print("Getting set up...")
    IMAGE_SIZE = 128
    NUM_CLASSES = 200
    image_dir, output_dir = get_dirs(FLAGS.image_dir, FLAGS.output_dir)
    if image_dir and output_dir:

        MODEL_FILE = os.path.join(output_dir, 'models/nn_20181014.ckpt')
        print("Building datasets from {} with batch size={}...".format(image_dir, FLAGS.batch_size))
        train_loader, test_loader = build_datasets(image_dir, FLAGS.batch_size)
        model = CNN(IMAGE_SIZE, NUM_CLASSES)
        
        print("Training the model with learning rate {}...".format(FLAGS.learning_rate))
        loss_list, acc_list = train(
            model, 
            train_loader, 
            FLAGS.epochs, 
            FLAGS.learning_rate, 
            FLAGS.display_every)

        print("Testing the model...")
        model.eval()
        test(model, test_loader)

        print("Saving the model to {}...".format(MODEL_FILE))
        torch.save(model.state_dict(), MODEL_FILE)

        print("Attempting a plot...")
        plot(loss_list, acc_list)

        print("Done!")


if __name__ == '__main__':
    # based on code from 
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--display_every', type=int, default=20,
                        help='Print status at intervals of this many steps.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    # parser.add_argument('--dropout', type=float, default=0.9,
    #                     help='Keep probability for training dropout.')
    parser.add_argument('--batch_size', type=float, default=100,
                        help='Batch size.')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='C:/datasets/CUB_200_2011/processed/data128/',
        help='Where to find the processed images')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='C:/Users/Leaf/Google Drive/School/BU-MET-CS-767/Project/birds/output',
        help='Where to store the output from this program')
    FLAGS, unparsed = parser.parse_known_args()
    main()