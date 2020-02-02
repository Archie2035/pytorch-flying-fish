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
from datetime import datetime
import argparse


time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

writer = SummaryWriter('runs/classifier_experiment_{}'.format(time_str))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Classifier')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=1, help='epoch')
    parser.add_argument('--model_output_path', default='./models/cifar10_model.pth', help='output model path')
    parser.add_argument('--show_plot', type=bool, default=False, help='show plot')

    args = parser.parse_args()
    return args


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


def solver(args):
    net = SimpleNet()
    show_info(net, device, args.show_plot)

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    bar = tqdm(range(args.epoch))
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

    torch.save(net.state_dict(), args.model_output_path)


if __name__ == '__main__':
    args = parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    torch.manual_seed(args.seed)

    print("config: {}".format(args))
    solver(args)
