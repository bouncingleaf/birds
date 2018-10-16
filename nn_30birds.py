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
import matplotlib.pyplot as plt

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
        # A print statement here helped me fish out a bad image file that was crashing my program!
        # print(img_name)
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
    """ 
    based on 
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset-e589682df0c5 
    https://cs230-stanford.github.io/pytorch-vision.html
    """
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes
        self.hidden_layer = 128
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64*32*32, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # Apply softmax here?
        # return F.log_softmax(out, dim=1)
        return out

def build_datasets(path, batch_size, validation_mode):
    train_csv = os.path.join(path, 'train/train_data.txt')
    train_dataset = BirdDataset(path + 'train/', csv_file=train_csv, transform=ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    if validation_mode:
        validation_csv = os.path.join(path, 'test/validation_data.txt')
        validation_dataset = BirdDataset(path + 'test/', csv_file=validation_csv, transform=ToTensor())
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, validation_loader
    else:
        test_csv = os.path.join(path, 'test/test_v_data.txt')
        test_dataset = BirdDataset(path + 'test/', csv_file=test_csv, transform=ToTensor())
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
            correct += (predicted == labels).sum().item()

        print('Accuracy on the test images: {} %'.format((correct / total) * 100))


def plot(loss_list, acc_list):
    p = plt.figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='ConvNet results')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), 
           np.array(acc_list) * 100,
           y_range_name='Accuracy',
           color='red')
    show(p)

def main(epochs, display_every, learning_rate, batch_size, validation_mode, model_file_id):
    print("Getting set up...")
    IMAGE_SIZE = 128
    NUM_CLASSES = 30

    HOME = True

    if HOME:
        image_dir = 'C:/datasets/Combined/processed/30birds128/'
        output_dir = 'C:/Users/Leaf/Google Drive/School/BU-MET-CS-767/Project/birds/output/'
        model_dir = 'C:/Users/Leaf/Google Drive/School/BU-MET-CS-767/Project/birds/models/'
    else: 
        image_dir = '/Users/Leaf/CS767/30birds128/'
        output_dir = '/Users/Leaf/CS767/birds/output/'
        model_dir = '/Users/Leaf/CS767/birds/models/'

    model_file = model_dir + model_file_id + "_nn_30birds.ckpt"
    print("Running with epochs={}, disp={}, learning_rate={}, batch_size={}\nimage_dir={}\n output_dir={}\n model_file={}"
    .format(epochs, display_every, learning_rate, batch_size, image_dir, output_dir, model_file))

    train_loader, test_or_validation_loader = build_datasets(image_dir, int(batch_size), validation_mode)
    model = CNN(IMAGE_SIZE, NUM_CLASSES)
    
    print("Training the model with learning rate {} for {} epochs...".format(learning_rate, epochs))
    loss_list, acc_list = train(
        model, 
        train_loader, 
        epochs, 
        learning_rate, 
        display_every)

    if validation_mode:
        print("Validating the model...")
    else:
        print("Testing the model...")
    model.eval()
    test(model, test_or_validation_loader)

    print("Saving the model to {}...".format(model_file))
    torch.save(model.state_dict(), model_file)

    print("Attempting a plot...")
    plot(loss_list, acc_list)

    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--display_every', type=int, default=100,
                        help='Print status at intervals of this many steps.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    # parser.add_argument('--dropout', type=float, default=0.9,
    #                     help='Keep probability for training dropout.')
    parser.add_argument('--batch_size', type=float, default=100,
                        help='Batch size.')
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS.epochs, FLAGS.display_every, FLAGS.learning_rate, FLAGS.batch_size, False, "20181016")