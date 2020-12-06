'''
@inproceedings{eccv18refrmv,
  title={Seeing deeply and bidirectionally: a deep learning approach for single image reflection removal},
  author={Yang, Jie and Gong, Dong and Liu, Lingqiao and Shi, Qinfeng},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={654--669},
  year={2018}
}
'''

import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.nn import init


###############################################################################
# Functions
###############################################################################
def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################
class Generator_cascade(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ns=[7,5,5], ngf=64, norm='batch', use_dropout=False, iteration=0, padding_type='zero', upsample_type='transpose'):
        super(Generator_cascade, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.iteration = iteration
        norm_layer = get_norm_layer(norm_type=norm)
        
        self.model1 = UnetGenerator(input_nc, output_nc, ns[0], ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.model2 = UnetGenerator(input_nc * 2, output_nc, ns[1], ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        if self.iteration > 0:
            self.model3 = UnetGenerator(input_nc * 2, output_nc, ns[2], ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    def forward(self, input, fromG2=False, pre_result=None):
        if fromG2 == False:
            x = self.model1(input)
            res = [x]
            for i in range(self.iteration + 1):
                if i % 2 == 0:
                    xy = torch.cat([x, input], 1)
                    z = self.model2(xy)
                    res += [z]
                else:
                    zy = torch.cat([z, input], 1)
                    x = self.model3(zy)
                    res += [x]
        else:
            res = [pre_result]
            for i in range(self.iteration + 1):
                if i % 2 == 0:
                    xy = torch.cat([pre_result, input], 1)
                    z = self.model2(xy)
                    res += [z]
                else:
                    zy = torch.cat([z, input], 1)
                    x = self.model3(zy)
                    res += [x]
        return res

class Generator_cascade_withEdge(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ns=[7,5,5], ngf=64, norm='batch', use_dropout=False, iteration=0, padding_type='zero', upsample_type='transpose'):
        super(Generator_cascade_withEdge, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.iteration = iteration
        norm_layer = get_norm_layer(norm_type=norm)
        
        self.model1 = UnetGenerator(input_nc +1, output_nc, ns[0], ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.model2 = UnetGenerator(input_nc * 2 +1, output_nc, ns[1], ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        if self.iteration > 0:
            self.model3 = UnetGenerator(input_nc * 2 +1, output_nc, ns[2], ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    def forward(self, input, fromG2=False, pre_result=None):
        if fromG2 == False:
            x = self.model1(input)
            res = [x]
            for i in range(self.iteration + 1):
                if i % 2 == 0:
                    xy = torch.cat([x, input], 1)
                    z = self.model2(xy)
                    res += [z]
                else:
                    zy = torch.cat([z, input], 1)
                    x = self.model3(zy)
                    res += [x]
        else:
            res = [pre_result]
            for i in range(self.iteration + 1):
                if i % 2 == 0:
                    xy = torch.cat([pre_result, input], 1)
                    z = self.model2(xy)
                    res += [z]
                else:
                    zy = torch.cat([z, input], 1)
                    x = self.model3(zy)
                    res += [x]
        return res



# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # currently support only input_nc == output_nc
        # assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True, use_dropout=use_dropout)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, outermost_input_nc=input_nc)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outermost_input_nc=-1):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if outermost and outermost_input_nc > 0:
            downconv = nn.Conv2d(outermost_input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        else:
            downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x1 = self.model(x)
        diff_h = x.size()[2] - x1.size()[2]
        diff_w = x.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2,
                        diff_h - diff_h // 2))
        if self.outermost:
            return x1
        else:
            return torch.cat([x1, x], 1)
