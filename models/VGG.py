import torch
import torch.nn as nn
from torchvision import models

class similarity_VGG_bn(nn.Module):
    def __init__(self, input_channels):
        super(similarity_VGG_bn, self).__init__()
        cnn_bn = models.vgg19_bn(pretrained=True).features.cuda()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU(inplace=True)
        
        for (sim,vgg) in zip(self.modules(),cnn_bn.modules()):
            if isinstance(sim,nn.Conv2d) or isinstance(sim,nn.BatchNorm2d):
                sim.weight = vgg.weight
                sim.bias = vgg.bias
    
    def forward(self, x):
        batch_size = x.size()[0]
        
        outputs=[]
        
        out = self.conv1(x)
        outputs.append(out)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        outputs.append(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
                
        out = self.conv3(out)
        outputs.append(out)
        out = self.bn3(out)
        out = self.relu3(out)
        
        out = self.conv4(out)
        outputs.append(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool4(out)
        
        out = self.conv5(out)
        outputs.append(out)
        out = self.bn5(out)
        out = self.relu5(out)
        
        return outputs
