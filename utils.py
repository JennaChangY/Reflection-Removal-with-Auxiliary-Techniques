import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

def save_img(img, path, gray=False):
    if gray == True:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(7.0/5,7.0/5) #dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.savefig(path, transparent=True, dpi=500, pad_inches = 0)
    
