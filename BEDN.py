#%%
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.optim as optim
import torch.utils.data as D

from torchvision import datasets,transforms

import math
import time
import random
import matplotlib.pyplot as plt
#import argparse

#%%
from fastai.vision.all import *

#%%
class SignActivation(Function):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input
# aliases
binarize = SignActivation.apply

# %%
class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.Tanh = nn.Tanh()

    def forward(self, input):
        output = self.Tanh(input).mul(2) #mul(2) for acceleration
        output = binarize(output)
        return output
        
class BinaryLinear(nn.Linear):
    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out

class BinConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=binarize(self.weight.org)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out

class BinConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *kargs, **kwargs):
        super(BinConvTranspose2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=binarize(self.weight.org)
        out = nn.functional.conv_transpose2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out


# %%
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.encoders = nn.Sequential(
            nn.Conv2d(3, 64 , kernel_size=3, padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(64 , 64 , kernel_size=3, padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(64 , 128 , kernel_size=3, stride=2 ,padding=[1,1]),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(128, 128 , kernel_size=3, padding=1),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(128, 256, kernel_size=3, stride=2, padding=[1,1]),
            nn.BatchNorm2d(256,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),
        )
        self.decoders = nn.Sequential(
            BinConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=[1,1]),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),
            
            BinConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=[1,1]),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(64, 11, kernel_size=3, padding=1),
            nn.BatchNorm2d(11,eps=1e-4, momentum=bnorm_momentum),
        )

    def forward(self, x):
        x = self.encoders(x)
        #x = x.view(-1, 256)
        x = self.decoders(x)
        return x

#%%
path = Path('./data/CamVid/')
codes = np.loadtxt(path/'codes.txt', dtype=str)

def label_func(fn): return f"{fn.parent}annot/{fn.stem}{fn.suffix}"

camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items = get_image_files,
                   get_y = label_func)

Trn_dls = camvid.dataloaders(path/"train", path=path, bs=2,num_workers=0)
Val_dls = camvid.dataloaders(path/"val", path=path, bs=2,num_workers=0)
Tst_dls = camvid.dataloaders(path/"test", path=path, bs=2,num_workers=0)

#%%
batch_size=768
test_batch_size=768
momentum = 0.9
decay = 0 #weight decay
damp= 0 #momentum dampening
lr = 0.1
epochs = 250
log_interval = test_batch_size
bnorm_momentum= 0.5 #0.4
update_list= [150,200,250]
# %%
model=Model()
learn=Learner(Trn_dls,model,loss_func=nn.CrossEntropyLoss(),opt_func=SGD, metrics=accuracy,cbs=CudaCallback)

#%%
epochs_trn=1
learn.model.cuda
#learn.dls.cuda
#%%
learn.fit(epochs_trn)
#learn.fit_one_cycle(1,0.01)