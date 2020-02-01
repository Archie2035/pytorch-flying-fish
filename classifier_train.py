import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from networks.simple_net import SimpleNet

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

torch.manual_seed(666)

writer = SummaryWriter('runs/classifier_experiment_1')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def show_img(img, show_plot=False, show_tensorboard=False):
    img = img * 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    if show_plot:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    if show_tensorboard:
        writer.add_image('train_images', img)


def show_info(net, device, show_plot=False):
    print("=========================================")
    print("device is: {}".format(device))
    print("net: {}".format(net))
    print("first mini_batch training data and labels")
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    label_info = [classes[i] for i in labels]
    print("label: {}".format(label_info))
    print("=========================================")
    writer.add_graph(net, images)
    show_img(torchvision.utils.make_grid(images), show_plot=show_plot, show_tensorboard=True)


def solver(epoch, model_output_path, show_plot):
    net = SimpleNet()
    show_info(net, device, show_plot)

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    bar = tqdm(range(epoch))
    start_time = time.time()

    for index in bar:
        i = 0
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            i += 1
            if i % 100 == 0:
                writer.add_scalar('training loss', loss, index * len(trainloader) + i)
                bar.set_description("epoch: {}, index: {}, loss: {}".format(index, i, loss))

    training_end_time = time.time()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    testing_end_time = time.time()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print("training elapsed: {}s , testing elapsed: {}s"
          .format(training_end_time - start_time, testing_end_time - start_time))

    torch.save(net.state_dict(), model_output_path)


if __name__ == '__main__':
    epoch = 1
    model_output_path = "./models/cifar10_model.pth"
    solver(epoch, model_output_path, show_plot=False)
