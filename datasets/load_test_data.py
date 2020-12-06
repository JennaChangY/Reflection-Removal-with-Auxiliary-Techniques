Jupyter Notebook
load_test_data.py
1 分鐘前
Python
File
Edit
View
Language
1
from torch.utils.data import Dataset, DataLoader
2
from PIL import Image
3
import numpy as np
4
import torch
5
import torch.utils.data
6
import torch.nn as nn
7
from torchvision import datasets, transforms
8
from torchvision.transforms import Compose, ToTensor, Resize
9
import os
10
from os.path import isfile, join
11
from glob import glob
12
​
13
​
14
class Load_Test_Data(Dataset):
15
    
16
    def __init__(self, root = '', transforms = None):
17
        
18
        self.I_list = []
19
        self.T_list = []
20
        self.transforms = transforms
21
    
22
        I_root = join(root, 'I_test')
23
        T_root = join(root, 'T_test')
24
        file_list = sorted(glob(join(I_root, '*.jpg')))
25
​
26
        for I_name in file_list:
27
            fname = I_name[len(I_root)+1:]
28
            T_name = join(T_root, fname)
29
            if not isfile(I_name) or not isfile(T_name):
30
                continue
31
            self.I_list += [I_name]
32
            self.T_list += [T_name]
33
            
34
        self.size = len(self.I_list)
35
​
36
    def __getitem__(self, index):
37
        
38
        index = index % self.size
39
        
40
        I = Image.open(self.I_list[index])
41
        T = Image.open(self.T_list[index])        
42
        
43
        if self.transforms is not None:
44
            I = self.transforms(I)
45
            T = self.transforms(T)
46
​
47
        return I, T
48
    
49
    def __len__(self):
50
        return self.size
51
    
52
​
