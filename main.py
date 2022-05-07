import math
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import SGD
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F

# normalize using 'zerocenter' normalization. i.e for each value do: (x-mean)/standard deviation
# i.e the same method used to normalize the normal distribution.
# the mean is now 1 and the standard deviation is 1.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0, 1)])

# We can tune the parameters for the loader later if needed

# Set to true after running once; this suppreses those annoying "files already downloaded and verified" messages
already_downloaded = False

training_batch = 16
test_batch = 2000
training_data = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             download=not already_downloaded,
                                             transform=transform)
train_loader = torch.utils.data.DataLoader(training_data,
                                           batch_size=training_batch,
                                           shuffle=True,
                                           num_workers=2)
test_data = torchvision.datasets.CIFAR10(root='./data',
                                         train=False,
                                         download=not already_downloaded,
                                         transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=test_batch,
                                          shuffle=False,
                                          num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.layer4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)
        self.layer5 = nn.Linear(512 * 8 * 8, 128)
        self.layer6 = nn.Linear(128, 64)
        self.layer7 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer6(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer7(x)
        return x

def train(epochs, net, test_every_epoch, loss_fn, optimizer):
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:  # print the loss after every 1000 mini-batches
                print("<==========================================================>")
                print(f'{i + 1:5d} images processed with loss: {running_loss / 1000:.3f}')
                print("<==========================================================>")
                print("\n")
                running_loss = 0
        print('epoch ', epoch + 1, i, ' complete.')
        if test_every_epoch:
            print("<==========================================================>")
            print("accuracy for epoch:")
            test(net)
            print("<==========================================================>")
    print('Finished Training')


def train_for_n_minutes(n, net, loss_fn, optimizer):
    start_time = time.time()
    end_time = time.time() + (n * 60)
    epoch = 0
    training_error = []
    test_error = []
    while end_time - time.time() > 0:
        print(f"Time elapsed: {((time.time() - start_time) / 60):3f}/{((end_time - start_time) / 60):3f} "
              f"(minutes)")
        epoch += 1
        running_loss = 0
        data_size = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("<==========================================================>")
        print(f'{data_size:5d} images processed with training loss: {running_loss / data_size:.3f}')
        training_error.append(running_loss / data_size)
        t_error = test(net)
        print(f"test loss for epoch:{t_error:.3f}")
        test_error.append(t_error)
        print('epoch ', epoch, i, ' complete.')
        print("<==========================================================>")
    print('Finished Training')
    plot_error_rates(training_error, test_error)


def plot_error_rates(training_error, test_error):
    x = [x for x in range(len(training_error))]
    plt.plot(x, training_error, label="training error")
    plt.plot(x, test_error, label="test error")
    plt.legend()
    plt.show()


def train_for_n_hours(n, net, loss_fn, optimizer):
    train_for_n_minutes(n * 60, net, loss_fn, optimizer)

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f'Network accuracy on {total} test images: {100 * correct // total} %')
    return 1 - (correct / total)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device1 = torch.device('cpu')
    print('Device:', device)

    net = Net().to(device)
    #optimizer = SGD(net.parameters(), lr=0.0001, momentum=0.9)
    optimizer = Adam(net.parameters(), lr=0.0001)
    train_for_n_minutes(20, net, loss_fn=nn.CrossEntropyLoss(),
                       optimizer=optimizer)
    test(net)
