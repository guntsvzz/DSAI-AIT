import torch
import cv2 # requires !pip3 install opencv-python
import os

from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

import pandas as pd
import matplotlib.pyplot as plt

from utils import matplotlib_imwrite

class SportDataset(Dataset):
    """Sport dataset."""

    def __init__(self, csv_file, root_dir, class_file, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            class_file (string): Path to the csv file with class names and indices.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        classes = pd.read_csv(class_file)
        self.class_dict = {row[1]:row[0] for i, row in classes.iterrows()}

        df = pd.read_csv(csv_file)
        if train:
            self.df = df[df['data set'] == 'train']
        else:
            self.df = df[df['data set'] == 'valid']

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        image = io.imread(img_name)

        if image.shape[-1] != 3:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image)

        label_keys = self.df.iloc[idx, 1]
        labels = self.class_dict[label_keys]
        labels = float(labels)

        sample = {'image': image, 'labels': labels}

        return sample

if __name__ == '__main__':
    train_transform = transforms.Compose(
        [
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
        ]
    )

    val_transform = transforms.transforms.Compose(
        [
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ]
    )

    csv_file = '/root/Datasets/Sports/sports.csv'
    class_file = '/root/Datasets/Sports/class_dict.csv'
    root_dir = '/root/Datasets/Sports'

    train_ds = SportDataset(csv_file=csv_file, class_file=class_file, root_dir=root_dir, train=True, transform=train_transform)
    val_ds = SportDataset(csv_file=csv_file, class_file=class_file, root_dir=root_dir, train=False, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    class_df = pd.read_csv(class_file, usecols=['class_index', 'class'])
    class_dict = { row[0]:row[1] for i, row in class_df.iterrows()}
    print("class dict: ", class_dict)

    dataiter = iter(train_loader)
    data = next(dataiter)

    plt.rcParams['figure.figsize'] = [15, 5]
    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(data['image'])
    matplotlib_imwrite(img_grid, 'sports-samples.png',one_channel=False)
    print(','.join(class_dict[data['labels'][j].item()] for j in range(8)))
