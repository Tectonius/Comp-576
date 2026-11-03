# -*- coding: utf-8 -*-
"""Assignment_2_Part_1_Cifar10_vp1.ipynb

Purpose: Implement image classsification nn the cifar10
dataset using a pytorch implementation of a CNN architecture (LeNet5)

Pseudocode:
1) Set Pytorch metada
- seed
- tensorboard output (logging)
- whether to transfer to gpu (cuda)

2) Import the data
- download the data
- create the pytorch datasets
    scaling
- create pytorch dataloaders
    transforms
    batch size

3) Define the model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates
        f. Calculate accuracy, other stats
    - Test:
        a. Calculate loss, accuracy, other stats

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop




"""

# Step 1: Pytorch and Training Metadata

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

#hyperparameters
batch_size = 128
epochs = 10
lr = 0.001
try_cuda = True
seed = 1000

# Architecture
num_classes = 10

#otherum
logging_interval = 10 # how many batches to wait before logging
logging_dir = None
grayscale = True

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/")
    runs_dir.mkdir(exist_ok = True)

    logging_dir = runs_dir / Path(f"{datetime_str}")

    logging_dir.mkdir(exist_ok = True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

#deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# downloading the cifar10 dataset


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x),
    transforms.ToTensor()
])


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def check_data_loader_dim(loader):
    # Checking the dataset
    for images, labels in loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

check_data_loader_dim(train_loader)
check_data_loader_dim(test_loader)

"""# 3) Creating the Model"""

layer_1_n_filters = 6
layer_2_n_filters = 16
fc_1_n_nodes = 120
kernel_size = 5
verbose = False
num_epochs = 10

# calculating the side length of the final activation maps
final_length = (32 - kernel_size + 2*2)//2 + 1
final_length = (final_length - kernel_size + 2*2)//2 + 1

if verbose:
    print(f"final_length = {final_length}")


class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, layer_1_n_filters, kernel_size=kernel_size, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layer_1_n_filters, layer_2_n_filters, kernel_size=kernel_size, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_length*final_length*layer_2_n_filters, fc_1_n_nodes),
            nn.Tanh(),
            nn.Linear(fc_1_n_nodes, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

model = LeNet5(num_classes=10, grayscale=True)

if cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters())

"""# Step 4: Train/Test Loop"""

# Defining the test and trainig loops

def train(epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()  # The choice of criterion depends on the output of the last layer of your network
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        logits,probas = model(data)

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        # log metrics
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()}')

    # 
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # The choice of criterion depends on the output of the last layer of your network
    
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            logits,probas  = model(data) 

            # Calculate the predicted label (highest probability)
            pred = probas.argmax(dim=1, keepdim=True)
            # Calculate the number of correct prediction and accuracy
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += criterion(logits, target).item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    writer.add_scalar('Test/Average loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)


for epoch in range(1, num_epochs + 1):
    train(epoch)
    test(epoch)

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""

