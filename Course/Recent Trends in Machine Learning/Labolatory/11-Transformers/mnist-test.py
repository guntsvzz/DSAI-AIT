import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from tqdm import tqdm
from vit import ViT

# Loading data
transform = ToTensor()

test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, shuffle=False, batch_size=16)

def test_ViT_classify(model, test_loader, device="cpu"):
    criterion = CrossEntropyLoss()
    correct, total = 0, 0
    test_loss = 0.0
    for batch in tqdm(test_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        
        y_hat = model(x)
        loss = criterion(y_hat, y) / len(x)
        test_loss += loss

        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).item()
        total += len(x)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")

# # Set the GPU Device
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)

#### Testing
model = ViT((1, 28, 28), n_patches=7, hidden_d=20, n_heads=2, out_d=10, device = device)
model = model.to(device)
model.load_state_dict(torch.load('vit-mnist-epoch-5.pth'))
model.eval()

test_ViT_classify(model, test_loader, device)