from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize
import os
from os.path import isfile, join
from glob import glob


class Load_Test_Data(Dataset):
    
    def __init__(self, root = '', transforms = None):
        
        self.I_list = []
        self.T_list = []
        self.transforms = transforms
    
        I_root = join(root, 'I_test')
        T_root = join(root, 'T_test')
        file_list = []
        file_list.extend(sorted(glob(join(I_root, '*.jpg'))))
        file_list.extend(sorted(glob(join(I_root, '*.png'))))
        self.file_list = file_list

        for I_name in file_list:
            fname = I_name[len(I_root)+1:]
            T_name = join(T_root, fname)
            if not isfile(I_name) or not isfile(T_name):
                continue
            self.I_list += [I_name]
            self.T_list += [T_name]
            
        self.size = len(self.I_list)

    def __getitem__(self, index):
        
        index = index % self.size
        
        I = Image.open(self.I_list[index])
        T = Image.open(self.T_list[index])        
        
        if self.transforms is not None:
            I = self.transforms(I)
            T = self.transforms(T)

        return I, T
    
    def __len__(self):
        return self.size
    
    def get_data_list(self):
        save_name_list = []
        for name in self.file_list:
            save_name_list.append(os.path.splitext(os.path.basename(name))[0])
        return save_name_list
