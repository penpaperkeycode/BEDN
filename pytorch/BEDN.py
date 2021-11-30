# https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55
# https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-training-3-4-8242d31de234
# https://wikidocs.net/57165
# https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html

#%%
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.optim as optim
import torch.utils.data as D
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import pathlib
from skimage.io import imread
from torchsummary import summary
import numpy as np

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

bnorm_momentum= 0.5
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
class CamvidDataset(D.Dataset):
    def __init__(self, 
                inputs: list, 
                labels: list, 
                transform = None
                ):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

        self.inputs_dtype = torch.float32
        self.labels_dtype = torch.float32


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx:int):
        input_ID = self.inputs[idx]
        label_ID = self.labels[idx]

        x, y = imread(input_ID), imread(label_ID)

        if self.transform is not None:
            x, y = self.transform(x, y)
        
        # y = y[np.newaxis, :]
        y_tmp = np.zeros([11, 360, 480])
        for i in range(11):
            if i == 0:
                continue
            y_tmp[i-1,:,:] = y == i
        
        y = y_tmp
        

        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.labels_dtype)

        x = torch.permute(x, [2,0,1])

        return x, y
#%%
# root directory
root = pathlib.Path.cwd() 
def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

#%%
inputs = get_filenames_of_path(root / 'data/CamVid/train')
labels = get_filenames_of_path(root / 'data/CamVid/trainannot')

# # random seed
# random_seed = 91
# g = torch.Generator()
# g.manual_seed(random_seed)

# dataset training
dataset_train = CamvidDataset(inputs=inputs,
                              labels=labels)
# dataloader training
dataloader_training = torch.utils.data.DataLoader(dataset=dataset_train,
                                 batch_size=6,
                                 shuffle=True)

#%%
inputs = get_filenames_of_path(root / 'data/CamVid/val')
labels = get_filenames_of_path(root / 'data/CamVid/valannot')

# dataset training
dataset_val = CamvidDataset(inputs=inputs,
                              labels=labels)
# dataloader training
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                 batch_size=6,
                                 shuffle=True,
                                 )

# %%
inputs = get_filenames_of_path(root / 'data/CamVid/test')
labels = get_filenames_of_path(root / 'data/CamVid/testannot')

# dataset training
dataset_test = CamvidDataset(inputs=inputs,
                              labels=labels)
# dataloader training
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                 batch_size=6,
                                 shuffle=False,
                                 )

#%% test
batch = dataset_train[0]
x, y = batch

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

#%%
model=Model()
model.cuda()
summary = summary(model, (3, 360, 480))

#%%
class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 test_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.test_DataLoader = test_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.mIoU = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate, self.mIoU

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            inputs, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(inputs)  # one forward pass  #inputs.permute(0, 3, 1, 2)
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
    
        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_mIoUs = [] # accumulate the mIoU here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)
                valid_mIoUs.append(self._mIoU(out, target))
                
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        mean_loss = np.mean(valid_losses)
        self.validation_loss.append(mean_loss)
        mean_mIoU = np.mean(valid_mIoUs)
        self.mIoU.append(mean_mIoU)

        # print('Validaton Mean Loss:{}'.format(mean_loss))
        # print('Validation Mean mIoU:{}'.format(mean_mIoU))

        batch_iter.close()

    def test(self):
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        test_losses = []  # accumulate the losses here
        test_mIoUs = [] # accumulate the mIoU here
        batch_iter = tqdm(enumerate(self.test_DataLoader), 'Validation', total=len(self.test_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                test_losses.append(loss_value)
                test_mIoUs.append(self._mIoU(out, target))
                
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        mean_loss = np.mean(test_losses)
        mean_mIoU = np.mean(test_mIoUs)
        print('Test Mean mIoU:{}'.format(mean_mIoU))
        print('Testset Mean Loss:{}'.format(mean_loss))

        batch_iter.close()
        return mean_mIoU, mean_loss

    def _mIoU(self, pred_mask, mask, smooth=1e-10, n_classes=11):
        with torch.no_grad():
            pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1)
            pred_mask = pred_mask.contiguous().view(-1)

            mask = torch.argmax(mask, dim=1)
            mask = mask.contiguous().view(-1)

            iou_per_class = []
            for clas in range(0, n_classes): #loop per pixel class
                true_class = pred_mask == clas
                true_label = mask == clas

                if true_label.long().sum().item() == 0: #no exist label in this loop
                    iou_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(true_class, true_label).sum().float().item()
                    union = torch.logical_or(true_class, true_label).sum().float().item()

                    iou = (intersect + smooth) / (union +smooth)
                    iou_per_class.append(iou)
            return np.nanmean(iou_per_class)
# %%
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

#%%
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


#%%
# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')

#%%
# criterion
criterion = torch.nn.CrossEntropyLoss()
# optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#%%
# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_val,
                  test_DataLoader=dataloader_test,
                  lr_scheduler=None,
                  epochs=400,
                  epoch=0,
                  notebook=None)
# %%
training_losses, validation_losses, lr_rates, mIoU = trainer.run_trainer()
print('Final Validaton Mean Loss:{}'.format(validation_losses[-1]))
print('Fianl Validation Mean mIoU:{}'.format(mIoU[-1]))
# %%
mean_mIoU, mean_loss = trainer.test()
