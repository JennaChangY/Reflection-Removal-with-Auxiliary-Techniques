from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import crop
import random
import os
from os.path import isfile, join
from glob import glob


def calculate_rand_size(width, height, scale_min, scale_max, limit):
    
    rand_scale = random.uniform(scale_min, scale_max)
    new_width = round(rand_scale * width)
    new_height = round(rand_scale * height)
    
    if new_width < limit or new_width < limit:
        rand_scale = max(limit/width, limit/height)
        new_width = round(rand_scale * width)
        new_height = round(rand_scale * height)
        
    width_space = width - new_width
    height_space = height - new_height
    
    starting_x = random.randint(0, height_space)
    starting_y = random.randint(0, width_space)
    
    return starting_x, starting_y, new_height, new_width

class Zhang_n_CEILNet(Dataset):
    
    def __init__(self, CEILNet_root = '', Zhang_root = '', transforms = None, Zhang_num_samples = 2000, CEILNet_num_samples=10000, Zhang_real=False):
        
        self.I_list = []
        self.R_list = []
        self.B_list = []
        self.transforms = transforms
        self.CEILNet_len = 0
        self.Zhang_syn_len = 0
        self.Zhang_len = 0
        
        # CEILNet
        file_list = sorted(glob(join(CEILNet_root, 'blended/*.png')))
        rand_idx = np.random.randint(len(file_list), size=CEILNet_num_samples)
        
        for idx in rand_idx:
            I_name = file_list[idx]
            B_name = I_name.replace('blended', 'transmission')
            R_name = I_name.replace('blended', 'reflection')
            if not isfile(I_name) or not isfile(B_name) or not isfile(R_name):
                continue
            self.I_list += [I_name]
            self.B_list += [B_name]
            self.R_list += [R_name]
            self.CEILNet_len += 1
            
        # Zhang syn
        file_list = sorted(glob(join(Zhang_root, 'synthetic/blended/*.png')))
        file_list.extend(sorted(glob(join(Zhang_root, 'synthetic_shuffle/blended/*.png'))))
        rand_idx = np.random.randint(len(file_list), size=Zhang_num_samples)
        
        for idx in rand_idx:
            I_name = file_list[idx]
            B_name = I_name.replace('blended', 'B_resize')
            R_name = I_name.replace('blended', 'R_resize')
            if not isfile(I_name) or not isfile(B_name) or not isfile(R_name):
                continue
            self.I_list += [I_name]
            self.B_list += [B_name]
            self.R_list += [R_name]
            self.Zhang_syn_len += 1
            
        # Zhang
        if Zhang_real:
            file_list = sorted(glob(join(Zhang_root, 'real/*/blended/*.jpg')))
            Zhang_I_list = []
            Zhang_B_list = []

            for I_name in file_list:
                B_name = I_name.replace('blended', 'transmission_layer')
                if not isfile(I_name) or not isfile(B_name):
                    continue
                Zhang_I_list += [I_name]
                Zhang_B_list += [B_name]
                self.Zhang_len += 1

            self.I_list.extend(Zhang_I_list)
            self.B_list.extend(Zhang_B_list)
            self.I_list.extend(Zhang_I_list)
            self.B_list.extend(Zhang_B_list)
            
        self.size = len(self.I_list)

    def __getitem__(self, index):
        
        index = index % self.size
        
        I = Image.open(self.I_list[index])
        B = Image.open(self.B_list[index])
        
        if index < (self.CEILNet_len+self.Zhang_syn_len):
            R = Image.open(self.R_list[index])
        elif index < (self.CEILNet_len+self.Zhang_syn_len+self.Zhang_len):
            R = B
        else:
            starting_x, starting_y, new_height, new_width = calculate_rand_size(I.size[0], I.size[1], 0.7, 0.9, 256)
            I = crop(I, starting_x, starting_y, new_height, new_width)
            B = crop(B, starting_x, starting_y, new_height, new_width)
            R = B
        
        if self.transforms is not None:
            I = self.transforms(I)
            R = self.transforms(R)
            B = self.transforms(B)
            
        return I, R, B
    
    def __len__(self):
        return self.size

class Zhang_data(Dataset):
    
    def __init__(self, root = '', limit=256, transforms = None):
        
        self.I_list = []
        self.B_list = []
        self.transforms = transforms
        self.limit = limit
    
        file_list = sorted(glob(join(root, '*/blended/*.jpg')))
        
        for I_name in file_list:
            B_name = I_name.replace('blended', 'transmission_layer')
            if not isfile(I_name) or not isfile(B_name):
                continue
            self.I_list += [I_name]
            self.B_list += [B_name]
        
        self.I_list.extend(self.I_list)
        self.B_list.extend(self.B_list)
            
        self.size = len(self.I_list)

    def __getitem__(self, index):
        
        index = index % self.size
        
        I = Image.open(self.I_list[index])
        B = Image.open(self.B_list[index])
        
        if index >= len(self.I_list) / 2:
            starting_x, starting_y, new_height, new_width = calculate_rand_size(I.size[0], I.size[1], 0.7, 0.9, self.limit)
            I = crop(I, starting_x, starting_y, new_height, new_width)
            B = crop(B, starting_x, starting_y, new_height, new_width)
        
        if self.transforms is not None:
            I = self.transforms(I)
            B = self.transforms(B)
            
        R = B
            
        return I, R, B
    
    def __len__(self):
        return self.size
    
class Zhang_test_data(Dataset):
    
    def __init__(self, root = '', transforms = None):
        
        self.I_list = []
        self.B_list = []
        self.transforms = transforms
    
        I_root = join(root, 'I_test')
        B_root = join(root, 'B_test')
        file_list = sorted(glob(join(I_root, '*.jpg')))

        for I_name in file_list:
            fname = I_name[len(I_root)+1:]
            B_name = join(B_root, fname)
            if not isfile(I_name) or not isfile(B_name):
                continue
            self.I_list += [I_name]
            self.B_list += [B_name]
            
        self.size = len(self.I_list)

    def __getitem__(self, index):
        
        index = index % self.size
        
        I = Image.open(self.I_list[index])
        B = Image.open(self.B_list[index])        
        
        if self.transforms is not None:
            I = self.transforms(I)
            B = self.transforms(B)
            
        R = B

        return I, R, B
    
    def __len__(self):
        return self.size
    
    
class CEILNet_data(Dataset):
    
    def __init__(self, root = '', transforms = None, num_samples = 10000):
        
        self.I_list = []
        self.B_list = []
        self.R_list = []
        self.transforms = transforms
    
        file_list = sorted(glob(join(root, 'blended/*.png')))
        rand_idx = np.random.randint(len(file_list), size=num_samples)
        
        for idx in rand_idx:
            I_name = file_list[idx]
            B_name = I_name.replace('blended', 'transmission')
            R_name = I_name.replace('blended', 'reflection')
            if not isfile(I_name) or not isfile(B_name) or not isfile(R_name):
                continue
            self.I_list += [I_name]
            self.B_list += [B_name]
            self.R_list += [R_name]
            
        self.size = len(self.I_list)

    def __getitem__(self, index):
        
        index = index % self.size
        
        I = Image.open(self.I_list[index])
        B = Image.open(self.B_list[index])
        R = Image.open(self.R_list[index])
        
#         if index >= len(self.I_list) / 2:
#             starting_x, starting_y, new_height, new_width = calculate_rand_size(I.size[0], I.size[1], 0.7, 0.9, 256)
#             I = crop(I, starting_x, starting_y, new_height, new_width)
#             B = crop(B, starting_x, starting_y, new_height, new_width)
        
        if self.transforms is not None:
            I = self.transforms(I)
            B = self.transforms(B)
            R = self.transforms(R)
            
        return I, R, B
    
    def __len__(self):
        return self.size
    