# -*- coding: utf-8 -*-
"""Assignment_2_Part_2_RNN_MNIST_vp1.ipynb
Overall structure:

1) Set Pytorch metada
- seed
- tensorflow output
- whether to transfer to gpu (cuda)

2) Import data
- download data
- create data loaders with batchsie, transforms, scaling

3) Define Model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop

# Step 1: Pytorch and Training Metadata
"""

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

# Best LSTM configuration found in the hidden-size sweep (99.15% test accuracy)
seed = 1000
epochs = 10
patience = 10  # stop after this many epochs without a new best test accuracy
batch_size = 64
test_batch_size = 1000
hidden_size = 512
optimizer_type = torch.optim.Adam
optimizer_kwargs = {'lr': 0.001}
try_cuda = True
run_name = 'lstm_h512_adam_lr0.001_ep10_bs64'
num_classes = 10
INPUT_SIZE = 28
criterion = nn.CrossEntropyLoss()
logging_interval = 10

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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),])

train_dataset = datasets.MNIST(root="/gpfs0/work/alex/Comp-576/assignment_2/data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="/gpfs0/work/alex/Comp-576/assignment_2/data", train=False, download=True, transform=transform)

if cuda:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

print(train_dataset.data.size()) # (60000, 28, 28)
print(train_dataset.targets.size()) # (60000)

"""# Step 3: Creating the Model"""

class RNNNet(nn.Module):
    def __init__(self):
        super(RNNNet, self).__init__()

        self.rnn = nn.RNN(input_size=INPUT_SIZE, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        logits = self.out(r_out[:, -1, :])
        probas = F.softmax(logits, dim=1)
        return logits, probas

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()

        self.rnn = nn.LSTM(input_size=INPUT_SIZE, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        logits = self.out(r_out[:, -1, :])
        probas = F.softmax(logits, dim=1)
        return logits, probas

class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()

        self.rnn = nn.GRU(input_size=INPUT_SIZE, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        logits = self.out(r_out[:, -1, :])
        probas = F.softmax(logits, dim=1)
        return logits, probas

model = LSTMNet()

if cuda:
    model.cuda()

optimizer = optimizer_type(model.parameters(), **optimizer_kwargs)

"""# Step 4: Train/Test Loop"""

# Defining the test and trainig loops

def train(epoch):
    model.train()

    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(-1, 28, 28)

        optimizer.zero_grad()
        logits,_ = model(data)

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(data)
        train_correct += logits.argmax(dim=1).eq(target).sum().item()

        # log metrics
        if batch_idx % logging_interval == 0:
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

    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, 28, 28)
            logits,probs  = model(data)
            pred = probs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += criterion(logits, target).item() * len(data)  # sum of per-sample losses

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
        print(f'Early stopping at epoch {epoch}: test accuracy has not improved for {patience} epochs '
              f'(best {best_accuracy:.2f}%)')
        writer.add_text('early_stopping', f'stopped at epoch {epoch} (patience {patience}, best accuracy {best_accuracy:.2f}%)')
        break
model.load_state_dict(best_state) # type: ignore
epoch_range = range(1, len(train_losses) + 1)
fig, (ax_acc, ax) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

ax_acc.plot(epoch_range, train_accuracies, color='tab:blue', label='Train')
ax_acc.plot(epoch_range, test_accuracies, color='tab:orange', label='Test')
ax_acc.axvline(best['epoch'], color='gray', linestyle='--', linewidth=1,
               label=f"Best epoch {best['epoch']} ({best['accuracy']:.1f}%)")
ax_acc.set_ylabel('Accuracy (%)')
ax_acc.set_title(f'{type(model.rnn).__name__} (hidden {hidden_size}) / {optimizer_type.__name__} {optimizer_kwargs}')
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

writer.close()
