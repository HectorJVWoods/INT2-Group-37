import torch 
import matplotlib.pyplot as plt  
import seaborn as sns  
import torch.nn as nn  
import torch.nn.functional as f
from torch.optim import SGD  
import torchvision

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Classes in the dataset as shown on https://www.cs.toronto.edu/~kriz/cifar.html
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


#We can tune the parameters for the loader later if needed

training_data = torchvision.datasets.CIFAR10(root='./data', 
                                             train=True, 
                                             download=True,
                                             transform = None)

train_loader = torch.utils.data.DataLoader(training_data,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=2)

test_data = torchvision.datasets.CIFAR10(root='./data', 
                                         train=False, 
                                         download=True,
                                         transform = None)

test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=False, 
                                          num_workers=2)
