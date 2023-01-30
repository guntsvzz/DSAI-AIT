# %%
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import copy
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,Subset
import matplotlib.pyplot as plt

# %%
def plot_data(ax,val_acc_history,loss_acc_history,fold):
    ax[0].plot(np.arange(25),np.array(val_acc_history),label = f"val acc model{fold}")
    ax[1].plot(np.arange(25),np.array(loss_acc_history),label = f"val acc model{fold}")
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Loss")    
    ax[0].set_title(f"Accuracy vs Epochs of model {fold+1}")
    ax[1].set_title(f"Loss vs Epochs of model {fold+1}")
    ax[0].legend()
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)   
# %%    
# Set device to GPU or CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Allow augmentation transform for training set, no augementation for val/test set

train_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

eval_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# %%
# load the data
dataset = datasets.ImageFolder(root='./chihuahua_muffin/', transform=train_preprocess)
train_dataloader = DataLoader(dataset, batch_size = 16) # get all 16 images into 1 batch

dataloaders = {'train': train_dataloader} 

from sklearn.model_selection import StratifiedKFold
from train_model import train_model
from senet_model import ResSENet18
import numpy as np

folds = 8
skf = StratifiedKFold(n_splits=folds, shuffle=True)
fig,ax = plt.subplots(1,2,sharex=True,figsize=(20,5))

model_acc = 0
for fold, (train_index, val_index) in enumerate(skf.split(dataset, dataset.targets)):
      batch_size = 4
      train = Subset(dataset, train_index)
      val = Subset(dataset, val_index)
      
      train_loader = DataLoader(train, batch_size=batch_size, 
                                                shuffle=True, num_workers=0, 
                                                pin_memory=False)
      val_loader = DataLoader(val, batch_size=batch_size, 
                                                shuffle=True, num_workers=0, 
                                                pin_memory=False)

      dataloaders = {'train': train_loader, 'val': val_loader}
      
      #There are only 2 classes
      model = ResSENet18()
      model.linear = nn.Linear(512,2)
      model.eval()
      model.to(device)

      dataloaders = {'train': train_loader, 'val': val_loader}
      criterion = nn.CrossEntropyLoss().to(device)
      optimizer =  optim.Adam(model.parameters(), lr = 0.005 + 0.005*2)

      bestmodel, val_acc_history, loss_acc_history = train_model(model, dataloaders, criterion, optimizer, device, 25, 'train_se_chimuffin')
 
      plot_data(ax,val_acc_history,loss_acc_history,fold)
      model_acc = model_acc + sum(val_acc_history)/len(val_acc_history)

plt.show()
print(f'Average accuracy of model: {model_acc/8}')