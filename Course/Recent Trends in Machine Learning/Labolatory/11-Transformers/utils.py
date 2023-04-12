import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def matplotlib_imwrite(img, filename, one_channel=False):
    matplotlib_imshow(img, one_channel)
    plt.savefig(filename)