import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
import torch.nn.functional as F
from train_module import train_model
import pickle

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

# Set up preprocessing of CIFAR-10 images to 3x32x32 with normalization
# using the magic ImageNet means and standard deviations. You can try
# RandomCrop, RandomHorizontalFlip, etc. during training to obtain
# slightly better generalization.

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Stds for CIFAR? std: 0.24703233 0.24348505 0.26158768
# Download CIFAR-10 and split into training, validation, and test sets

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=preprocess)

# Split the training set into training and validation sets randomly.
# CIFAR-10 train contains 50,000 examples, so let's split 80%/20%.

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])

# Download the test set. If you use data augmentation transforms for the training set,
# you'll want to use a different transformer here.

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=preprocess)

# Dataset objects are mainly designed for datasets that can't fit entirely into memory.
# Dataset objects don't load examples into memory until their __getitem__() method is
# called. For supervised learning datasets, __getitem__() normally returns a 2-tuple
# on each call. To make a Dataset object like this useful, we use a DataLoader object
# to optionally shuffle then batch the examples in each dataset. During training.
# To keep our memory utilization small, we'll use 4 images per batch, but we could use
# a much larger batch size on a dedicated GPU. To obtain optimal usage of the GPU, we
# would like to load the examples for the next batch while the current batch is being
# used for training. DataLoader handles this by spawining "worker" threads that proactively
# fetch the next batch in the background, enabling parallel training on the GPU and data
# loading/transforming/augmenting on the CPU. Here we use num_workers=2 (the default)
# so that two batches are always ready or being prepared.

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                               shuffle=True, num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                             shuffle=False, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                              shuffle=False, num_workers=2)

dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

# Device 'cuda' or 'cuda:0' means GPU slot 0.
# If you have more than one GPU, you can select other GPUs using 'cuda:1', 'cuda:2', etc.
# In terminal (Linux), you can check memory using in each GPU by using command
# $ nvidia-smi
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Using device', device)

#create pretrained googlenet instance
model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True, aux_logits = True)
model.aux1.fc2 = nn.Linear(1024,10)
model.aux2.fc2 = nn.Linear(1024,10)
model.fc = nn.Linear(1024,10)
model = model.to(device)

# CrossEntropyLoss for multinomial classification (because we have 10 classes)
criterion = nn.CrossEntropyLoss()
params_to_update = model.parameters()
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

#Train!
best_model, val_acc_history, loss_acc_history = train_model(model, dataloaders, criterion, optimizer, device, 10, 'googlenet_pre_module_lr_0.001_bestsofar',is_inception=True)

#Save training statistics 
with open('googlenet_pre_module_lr_0.001_bestsofar.pkl','wb') as f:
    pickle.dump({'best model' : best_model,'val_acc_history': val_acc_history,'loss_acc_history':loss_acc_history},f)

from test_module import test_model

model.load_state_dict(torch.load('googlenet_pre_module_lr_0.001_bestsofar.pth'))
test_dataloaders = { 'test': test_dataloader }
test_acc, test_loss = test_model(model, test_dataloaders, criterion, device)
