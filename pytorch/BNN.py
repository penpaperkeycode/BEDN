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

# %%
def timeSince(since):
    now = time.time()
    s = now - since
    return s
# %%
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64 , kernel_size=3, padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(64 , 64 , kernel_size=3, padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            nn.MaxPool2d(kernel_size=2),

            BinConv2d(64 , 128 , kernel_size=3, padding=1),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),


            BinConv2d(128, 128 , kernel_size=3, padding=1),
            nn.BatchNorm2d(128,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            nn.MaxPool2d(kernel_size=2),

            BinConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            BinConv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=bnorm_momentum),
            BinaryTanh(),

            nn.MaxPool2d(kernel_size=2),
            nn.AvgPool2d(kernel_size=4)
        )
        self.classifier = nn.Sequential(
            BinaryLinear(256, 10),
            nn.BatchNorm1d(10,momentum=bnorm_momentum),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256)
        x = self.classifier(x)
        return x

# %%
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
train_loader = D.DataLoader(datasets.CIFAR10('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),batch_size=batch_size, shuffle=True)

val_loader = D.DataLoader(datasets.CIFAR10('./data', train=False, download=True,
                    transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),batch_size=test_batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model=Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                        lr=lr, momentum=momentum, weight_decay=decay, dampening= damp)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=update_list, gamma=0.1)

# %%
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        
        optimizer.step()
        
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        
        if batch_idx % log_interval == 0:
            tlos.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#%%
def validate():
    global best_acc
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        if acc > best_acc:
            best_acc = acc
            #save_state(model, best_acc)
            #torch.save(model,state_dict(),PATH)
        
        accur.append( 100.*correct/total)
        print('Validation Epoch:', epoch, '\t\tLoss: %.5f | Acc: %.3f%% (%d/%d)'% (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Best Accuracy:: {:.2f} %'.format(best_acc))
        
# %%
time_graph=[]
e=[]
accur=[]
tlos=[]
best_acc=0
start = time.time()
# %%
if torch.cuda.is_available():
    print("CUDA Available!")
    model.cuda()
else:
    print("Only CPU!")

for epoch in range(1, epochs + 1):
    e.append(epoch)
    train(epoch)   
    seco=timeSince(start)
    time_graph.append(seco)
    validate()
    scheduler.step()

# %%
#fianl binarize
for p in list(model.parameters()):
    p.data[p.data >= 0] = 1
    p.data[p.data < 0] = -1
    
print(model)
print(model.features[0]._parameters)

#%%
print(time_graph)
plt.title('Training for CIFAR10 with epoch', fontsize=20)
plt.ylabel('time (s)')
plt.plot(e,time_graph)
plt.show()
plt.title('Accuracy With epoch', fontsize=20)
plt.plot(e,accur)
plt.show()
plt.title('Validation loss With epoch', fontsize=20)
plt.plot(tlos)
plt.show()

# %%
