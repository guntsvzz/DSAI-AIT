import time
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
import torch.nn.functional as F



def test_model(model, dataloaders, criterion, device):
    '''
    train_model function

    Train a PyTorch model for a given number of epochs.
    
            Parameters:
                    model: Pytorch model
                    dataloaders: dataset
                    criterion: loss function
                    optimizer: update weights function
                    num_epochs: number of epochs
                    weights_name: file name to save weights
                    is_inception: The model is inception net (Google LeNet) or not

            Returns:
                    model: Best model from evaluation result
                    val_acc_history: evaluation accuracy history
                    loss_acc_history: loss value history
    '''
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
            
    for inputs, labels in dataloaders['test']:

        # Inputs is one batch of input images, and labels is a corresponding vector of integers
        # labeling each image in the batch. First, we move these tensors to our target device.
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # Gather our summary statistics        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(dataloaders['test'].dataset)
    test_acc = running_corrects.double() / len(dataloaders['test'].dataset)
    test_end = time.time()

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', test_loss, test_acc))

    return test_acc, test_loss