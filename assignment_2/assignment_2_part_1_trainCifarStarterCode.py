# -*- coding: utf-8 -*-
"""Assignment_2_Part_1_Cifar10_vp1.ipynb

Purpose: Implement image classification on the cifar10
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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # no display on Slurm compute nodes
import matplotlib.pyplot as plt

# Best configuration found in the hyperparameter sweeps (65.3% test accuracy)
seed = 1000
epochs = 100
patience = 10  # stop after this many epochs without a new best test accuracy
batch_size = 64
activation_func = nn.Tanh()
optimizer_type = torch.optim.Adam
optimizer_kwargs = {'lr': 0.001, 'weight_decay': 0.0}
normalize = True
augment = True
batchnorm = True
dropout = 0.3
conv_filters = (16, 32)
grayscale = True
try_cuda = True
run_name = 'tanh_adam_lr0.001_bs64_norm_aug_f16-32_bn_do0.3'

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

logging_dir = Path("runs") / f"{datetime_str}_{run_name}"
logging_dir.mkdir(parents=True, exist_ok=True)
logging_dir = str(logging_dir.absolute())
writer = SummaryWriter(log_dir=logging_dir)
torch.manual_seed(seed)
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False

"""# Step 2: Data Setup"""

# downloading the cifar10 dataset

train_mean, train_std = 0.4822, 0.2379

to_gray = transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x)
to_tensor = [transforms.ToTensor()]
if normalize:
    to_tensor.append(transforms.Normalize((train_mean,), (train_std,)))

augmentation = [transforms.RandomCrop(28, padding=2), transforms.RandomHorizontalFlip()] if augment else []

train_transform = transforms.Compose([to_gray] + augmentation + to_tensor)
test_transform = transforms.Compose([to_gray] + to_tensor)

train_dataset = datasets.ImageFolder('./CIFAR10/Train', transform=train_transform)
test_dataset = datasets.ImageFolder('./CIFAR10/Test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""# 3) Creating the Model"""

layer_1_n_filters, layer_2_n_filters = conv_filters
fc_1_n_nodes = 120
fc_2_n_nodes = 84
kernel_size = 5
conv_1_padding = 2
conv_2_padding = 0
image_side = train_dataset[0][0].shape[-1]
final_length = (image_side - kernel_size + 2*conv_1_padding + 1) // 2
final_length = (final_length - kernel_size + 2*conv_2_padding + 1) // 2

class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        def conv_block(c_in, c_out, padding):
            return ([nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding)]
                    + ([nn.BatchNorm2d(c_out)] if batchnorm else [])
                    + [activation_func, nn.MaxPool2d(kernel_size=2, stride=2)])

        self.features = nn.Sequential(
            *conv_block(in_channels, layer_1_n_filters, conv_1_padding),
            *conv_block(layer_1_n_filters, layer_2_n_filters, conv_2_padding)
        )

        drop = [nn.Dropout(dropout)] if dropout > 0 else []
        self.classifier = nn.Sequential(
            nn.Linear(final_length*final_length*layer_2_n_filters, fc_1_n_nodes),
            activation_func,
            *drop,
            nn.Linear(fc_1_n_nodes, fc_2_n_nodes),
            activation_func,
            *drop,
            nn.Linear(fc_2_n_nodes, num_classes)
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

optimizer = optimizer_type(model.parameters(), **optimizer_kwargs)

"""# Step 4: Train/Test Loop"""

# Defining the test and trainig loops

def train(epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()  # The choice of criterion depends on the output of the last layer of your network
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        logits,_ = model(data)

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(data)
        train_correct += logits.argmax(dim=1).eq(target).sum().item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()}') # type: ignore

    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    train_loss /= len(train_loader.dataset) # type: ignore
    train_accuracy = 100. * train_correct / len(train_loader.dataset) # type: ignore
    writer.add_scalar('Train/Average loss', train_loss, epoch)
    writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
    return train_loss, train_accuracy

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # The choice of criterion depends on the output of the last layer of your network

    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            logits,probs  = model(data) 

            pred = probs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += criterion(logits, target).item() * len(data)

    test_loss /= len(test_loader.dataset) # type: ignore
    accuracy = 100. * correct / len(test_loader.dataset) # type: ignore
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n') # type: ignore
    writer.add_scalar('Test/Average loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)
    return test_loss, accuracy


train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []
best = {'epoch': 0, 'test_loss': float('inf'), 'accuracy': 0.0}
best_state = None
best_accuracy, epochs_without_improvement, stopped_epoch = 0.0, 0, None
for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_loss, accuracy = test(epoch)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    if test_loss < best['test_loss']:
        best = {'epoch': epoch, 'test_loss': test_loss, 'accuracy': accuracy}
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if accuracy > best_accuracy:
        best_accuracy, epochs_without_improvement = accuracy, 0
    else:
        epochs_without_improvement += 1
    if patience and epochs_without_improvement >= patience:
        stopped_epoch = epoch
        print(f'Stopped early at {epoch}: test accuracy has degraded for {patience} epochs'
              f'(best {best_accuracy:.2f}%)')
        writer.add_text('early_stopping', f'stopped at epoch {epoch} (patience: {patience}, best accuracy: {best_accuracy:.2f}%)')
        break
model.load_state_dict(best_state) #type: ignore
epoch_range = range(1, len(train_losses) + 1)
fig, (ax_acc, ax) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

ax_acc.plot(epoch_range, train_accuracies, color='tab:blue', label='Train')
ax_acc.plot(epoch_range, test_accuracies, color='tab:orange', label='Test')
ax_acc.axvline(best['epoch'], color='gray', linestyle='--', linewidth=1,
               label=f"Best epoch {best['epoch']} ({best['accuracy']:.1f}%)")
ax_acc.set_ylabel('Accuracy (%)')
arch = f'filters {layer_1_n_filters}/{layer_2_n_filters}{" BN" if batchnorm else ""}{f" dropout {dropout}" if dropout else ""}'
ax_acc.set_title(f'{type(activation_func).__name__} / {optimizer_type.__name__} {optimizer_kwargs}\n{arch}', fontsize=10)
ax_acc.legend()
ax_acc.grid(alpha=0.3)

ax.plot(epoch_range, train_losses, label='Train')
ax.plot(epoch_range, test_losses, label='Test')
ax.axvline(best['epoch'], color='gray', linestyle='--', linewidth=1,
           label=f"Best epoch {best['epoch']} ({best['accuracy']:.1f}%)")
ax.set_xlabel('Epoch')
ax.set_ylabel('Average cross entropy loss')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
loss_plot_path = os.path.join(logging_dir, 'loss_curves.png')
fig.savefig(loss_plot_path, dpi=150)
plt.close(fig)

# visualization

conv1_weights = model.features[0].weight.detach().cpu().squeeze(1) # type: ignore
m = conv1_weights.abs().max()
n_cols = 3 if len(conv1_weights) <= 6 else 8
n_rows = -(-len(conv1_weights) // n_cols)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(7 if n_cols == 3 else 10, 2.25 * n_rows), squeeze=False)
for i, ax in enumerate(axs.flat):
    ax.axis('off')
    if i < len(conv1_weights):
        ax.imshow(conv1_weights[i].numpy(), cmap='gray', vmin=-m, vmax=m, interpolation='nearest')
fig.tight_layout()
conv1_plot_path = os.path.join(logging_dir, 'conv1_weights.png')
fig.savefig(conv1_plot_path, dpi=150)
plt.close(fig)

n_conv = sum(isinstance(layer, nn.Conv2d) for layer in model.features)
act_sum = [0.] * n_conv
act_sq_sum = [0.] * n_conv
act_count = [0] * n_conv
model.eval()
with torch.no_grad():
    for data, _ in test_loader:
        if cuda:
            data = data.cuda()
        x = data
        i = 0
        for layer in model.features:
            x = layer(x)
            if layer is activation_func:
                a = x.transpose(0, 1).reshape(x.size(1), -1).double()  # (n_filters, batch*H*W)
                act_sum[i] = act_sum[i] + a.sum(dim=1)
                act_sq_sum[i] = act_sq_sum[i] + (a ** 2).sum(dim=1)
                act_count[i] += a.size(1)
                i += 1

for i, layer_name in enumerate(f'conv{k + 1}' for k in range(n_conv)):
    act_mean = act_sum[i] / act_count[i]
    act_std = (act_sq_sum[i] / act_count[i] - act_mean ** 2).clamp(min=0).sqrt()
    act_mean, act_std = act_mean.cpu().numpy(), act_std.cpu().numpy()
    n_filters = len(act_mean)
    filter_ids = range(1, n_filters + 1)

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * n_filters + 2), 4))
    ax.bar(filter_ids, act_mean, yerr=act_std, capsize=4, color='tab:blue',
           ecolor='black', label='Mean (error bar = ±1 std)')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xticks(list(filter_ids))
    ax.set_xlabel('Filter number')
    ax.set_ylabel(f'Activation ({type(activation_func).__name__} output)')
    ax.set_title(f'{layer_name}: activation mean ± std per filter ({n_filters} filters, test set)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, f'{layer_name}_activation_stats.png'), dpi=150)
    plt.close(fig)
    for f in range(n_filters):
        writer.add_scalar(f'Activations/{layer_name}_mean', act_mean[f], f + 1)
        writer.add_scalar(f'Activations/{layer_name}_std', act_std[f], f + 1)

writer.close()
