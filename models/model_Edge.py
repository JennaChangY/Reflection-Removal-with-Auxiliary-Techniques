import torch
import torch.nn as nn
from torch.autograd import Variable 

class Edge_UNet(nn.Module):
    def __init__(self):
        super(Edge_UNet, self).__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64,128,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,128,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128,256,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(256,512,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        self.downsample5 = nn.Sequential(
            nn.Conv2d(512,512,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding= 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding= 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding= 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding= 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding= 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding= 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding= 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.upsample5 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding= 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding= 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding= 1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.downsample1(x)
        #print('ds1',x.size())
        skip1 = x
        x = self.downsample2(x)
        #print('ds2',x.size())
        skip2 = x
        x = self.downsample3(x)
        #print('ds3',x.size())
        skip3 = x
        x = self.downsample4(x)
        #print('ds4',x.size())
        skip4 = x
        x = self.downsample5(x)
        #print('ds5',x.size())


        x = self.upsample1(x)
        #print('us1',x.size())
        x = x + skip4
        x = self.upsample2(x)
        #print('us2',x.size())
        x = x + skip3
        x = self.upsample3(x)
        #print('us3',x.size())
        #x = x + skip2
        x = self.upsample4(x)
        #print('us4',x.size())
        #x = x + skip1
        x = self.upsample5(x)
        #print('us5',x.size())
        x = self.output(x)
        #print('out',x.size())
        return x
