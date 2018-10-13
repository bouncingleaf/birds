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

# Constants
HOME = '/Users/leaf/CS767/data128/'
TRAIN_PATH = os.path.join(HOME, 'train/')
TEST_PATH = os.path.join(HOME, 'test/')
PIXELS = 128 

# Hyperparameters
num_epochs = 10
num_classes = 200
batch_size = 100
learning_rate = 0.001

MODEL_STORE_PATH = os.path.join(HOME, 'pytorch_models/')

class BirdDataset(Dataset):
    def __init__(self, csv_file, transform=None):

        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.birds = pd.read_csv(csv_file, sep=' ', names=['path','class_id'])
        self.transform = transform

    def __len__(self):
        return len(self.birds)

    def __getitem__(self, idx):
        img_name = self.birds.iloc[idx, 0]
        img = Image.open(img_name)
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

print("Building datasets...")
TRAIN_CSV_FILE = os.path.join(TRAIN_PATH, 'train_data.txt')
TEST_CSV_FILE = os.path.join(TEST_PATH, 'test_data.txt')
train_dataset = BirdDataset(csv_file=TRAIN_CSV_FILE, transform=ToTensor())
test_dataset = BirdDataset(csv_file=TEST_CSV_FILE, transform=ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    """ based on https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset-e589682df0c5 """
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(PIXELS*PIXELS*2, PIXELS*2)
        self.fc2 = nn.Linear(PIXELS*2, 200)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

print("Initializing model")
model = CNN()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print("Training the model")
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

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data in test_loader:
        outputs = model(data['image'])
        _, predicted = torch.max(outputs.data, 1)
        labels = data['label']
        total += labels.size(0)
        print(predicted, labels)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model2.ckpt')

p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
show(p)
