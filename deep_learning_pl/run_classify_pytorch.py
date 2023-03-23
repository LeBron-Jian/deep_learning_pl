import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307), (0.3081))
     ])
batch_size = 64
# Data set
train_dataset = torchvision.datasets.MNIST(root='./datasets',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./datasets',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
dataiter = iter(train_loader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 1000

for epoch in range(num_epochs):
    sum_loss = 0.0
    for i, data in enumerate(train_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = F.softmax(net(images))
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate loss
        sum_loss += loss.item()
        if i % 100 == 99:
            print('[%d,%d] loss:%.03f' % (epoch + 1, i + 1, sum_loss / 100))
            sum_loss = 0.0
net.eval()
correct = 0
total = 0
for data_test in test_loader:
    images, labels = data_test
    images = images.to(device)
    labels = labels.to(device)
    output_test = net(images)
    _, predicted = torch.max(output_test, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum()
print("correct1: ", correct)
print("Test acc: {0}".format(correct.item() / len(test_dataset)))
