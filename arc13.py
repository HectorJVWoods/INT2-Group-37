import os
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import math
import numpy as np
import torch
import tarfile
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import SGD
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
from torch.utils.data import random_split

# normalize using 'zerocenter' normalization. i.e for each value do: (x-mean)/standard deviation
# i.e the same method used to normalize the normal distribution.
# the mean is now 1 and the standard deviation is 1.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0, 1)])

augment = transforms.Compose(
    [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), transforms.ToTensor(), transforms.Normalize(0, 1)])

# Classes in the dataset as shown on https://www.cs.toronto.edu/~kriz/cifar.html
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# We can tune the parameters for the loader later if needed

# Set to true after running once; this suppreses those annoying "files already downloaded and verified" messages
already_downloaded = True

training_batch = 128
test_batch = 1000
training_data = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             download=not already_downloaded,
                                             transform=transform)

augmented_data = torchvision.datasets.CIFAR10(root='./data',
                                              train=True,
                                              download=not already_downloaded,
                                              transform=augment)

training_data = torch.utils.data.ConcatDataset([training_data, augmented_data])

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
        self.layer3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.layer5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer6 = nn.Linear(8192, 128)
        self.layer7 = nn.Linear(128, 64)
        self.layer8 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.soft_max = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.layer6(x)
        x = F.relu(x)
        x = self.layer7(x)
        x = F.relu(x)
        x = self.layer8(x)
        # x = self.soft_max(x)
        return x


def save_model(path, net_state):
    torch.save(net_state, path)


def load_model(path):
    loaded_net = Net()
    loaded_net.load_state_dict(torch.load(path))
    return loaded_net


def train_for_n_minutes(n, net, loss_fn, optimizer, file_path, show_graph):
    diff = []
    training_error = []
    correct = 0
    total = 0
    start_time = time.time()
    end_time = time.time() + (n * 60)
    epoch = 0
    training_loss = []
    test_error = []
    best_params_so_far = net.state_dict()
    best_epoch = -1
    lowest_error = math.inf
    f1_scores = []
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("<==========================================================>")
        print(f'{i} images processed with training loss: {running_loss / data_size:.3f}')
        training_loss.append(running_loss / data_size)
        training_error.append(1 - (correct / total))
        t_error, f1_score = test(net)
        f1_scores.append(f1_score)
        print(f'training accuracy:  {100 * (correct / total)}%')
        print(f"test loss for epoch:{t_error:.3f}")
        test_error.append(t_error)
        if t_error < lowest_error:
            print("New record for test error.")
            best_params_so_far = net.state_dict()
            lowest_error = t_error
            best_epoch = epoch
        diff.append((1 - (correct / total)) - t_error)
        print('epoch ', epoch, ' complete.')
        print("<==========================================================>")
    print('Finished Training')
    print(
        f"Best parameters were at epoch {best_epoch}, With test error rate {lowest_error}.")
    if show_graph:
        print(f"Saving these parameters to {file_path}")
        save_model(file_path, best_params_so_far)
        plot_error_rates(training_error, test_error, diff, f1_scores)
    return best_params_so_far, lowest_error, best_epoch, training_error, test_error


def plot_error_rates(training_error, test_error, diff, f1_scores):
    x = [x for x in range(len(training_error))]
    # plt.plot(x, training_loss, label="training loss")
    plt.plot(x, test_error, label="test error")
    plt.plot(x, training_error, label="training error")
    plt.plot(x, diff, label="test-training Difference")
    plt.plot(x, f1_scores, label="f1-score")
    plt.legend()
    plt.show()


def plot_multiple_error_rates(tr_errors, te_errors, labels, label_name):
    if len(tr_errors) != len(te_errors) or len(tr_errors) != len(labels):
        print("Dim. are not the same size!")
        return
    for i in range(len(tr_errors)):
        training_error = tr_errors[i]
        test_error = te_errors[i]
        l = labels[i]
        x = [x for x in range(len(training_error))]
        plt.plot(x, training_error, label="train (" + label_name + "=" + str(l) + ")")
        plt.plot(x, test_error, label="test (" + label_name + "=" + str(l) + ")")
    plt.legend()
    plt.show()


def train_for_n_hours(n, net, loss_fn, optimizer, file_path):
    train_for_n_minutes(n * 60, net, loss_fn, optimizer, file_path)


def optimize_batches(mins_per_train_cycle, file_path, loss_fn, optim, learn_rate):
    global train_loader, test_loader
    batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    best_params = None
    lowest_error = math.inf
    test_errors = []
    test_graphs = []
    train_graphs = []
    best_epoch = -1
    best_bs = None
    for BS in batch_sizes:
        print("----------------------------------")
        print("----------------------------------")
        print(f"Trying batch size: {BS}")
        print("----------------------------------")
        print("----------------------------------")
        train_loader, test_loader = init_data_sets(train_batch_size=BS, test_batch_size=1000)
        t_net = Net().to(device)
        params, test_error, epoch, train_graph, test_graph = train_for_n_minutes(
            mins_per_train_cycle, t_net, loss_fn(), optim(lr=learn_rate, params=t_net.parameters()), "", False)
        test_errors.append(test_error)
        train_graphs.append(train_graph)
        test_graphs.append(test_graph)
        if test_error < lowest_error:
            print("New record for test error!")
            best_params = net.state_dict()
            lowest_error = test_error
            best_epoch = epoch
            best_bs = BS
    print("----------------------------------")
    print("----------------------------------")
    print(f"Optimization complete. The optimal batch size was {best_bs}, with a test error of {lowest_error}, "
          f"at an optimal epoch of {best_epoch}")
    print(f"Saving the best model to {file_path}")
    print(test_errors)
    save_model(file_path, best_params)
    plot_multiple_error_rates(train_graphs, test_graphs, batch_sizes, "bs")
    return best_bs, best_params


def optimize_learning_rates(mins_per_train_cycle, file_path, loss_fn, optim):
    # learning_rates = [0.005, 0.001, 0.0005]
    learning_rates = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    best_params = None
    lowest_error = math.inf
    test_errors = []
    test_graphs = []
    train_graphs = []
    best_epoch = -1
    best_lr = None
    for LR in learning_rates:
        print("----------------------------------")
        print("----------------------------------")
        print(f"Trying learning rate: {LR}")
        print("----------------------------------")
        print("----------------------------------")
        t_net = Net().to(device)
        params, test_error, epoch, train_graph, test_graph = train_for_n_minutes(
            mins_per_train_cycle, t_net, loss_fn(), optim(lr=LR, params=t_net.parameters()), "", False)
        test_errors.append(test_error)
        train_graphs.append(train_graph)
        test_graphs.append(test_graph)
        if test_error < lowest_error:
            print("New record for test error!")
            best_params = net.state_dict()
            lowest_error = test_error
            best_epoch = epoch
            best_lr = LR
    print("----------------------------------")
    print("----------------------------------")
    print(f"Optimization complete. The optimal learning rate was {best_lr}, with a test error of {lowest_error}, "
          f"at an optimal epoch of {best_epoch}")
    print(f"Saving the best model to {file_path}")
    print(test_errors)
    save_model(file_path, best_params)
    plot_multiple_error_rates(train_graphs, test_graphs, learning_rates, "lr")
    return best_lr, best_params


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
    f1_score = F1(correct, (total - correct))
    print(f'Network accuracy on {total} test images: {100 * correct / total:.2f} %')
    print(f' F1: {f1_score:.3f}')
    return 1 - (correct / total), f1_score


def F1(TP, FN):
    precision = 1
    recall = TP / (TP + FN)
    return 2 * ((precision * recall) / (precision + recall))


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device1 = torch.device('cpu')
    print('Device:', device)

    net = Net().to(device)
    # net = load_model("model1").to(device)
    train_for_n_minutes(1, net, loss_fn=nn.CrossEntropyLoss(),
                        optimizer=SGD(net.parameters(), lr=0.01, momentum=0.9), file_path="model1",
                        show_graph=True)
# torch.optim.Adam(lr=0.0001, params=net.parameters())
# optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimize_batches(20, "model2", nn.CrossEntropyLoss, torch.optim.Adam, 0.0005)