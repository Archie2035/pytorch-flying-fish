import torch
from networks.simple_net import SimpleNet
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

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

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def show_img(img, label, predict, elapsed_time):
    img = img * 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.title("label:{}, predict:{}, elapsed_time:{}ms".format(classes[label], classes[predict], elapsed_time))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def solver(model_output_path):
    net = SimpleNet()
    net.load_state_dict(torch.load(model_output_path))
    net.to(device)

    with torch.no_grad():
        for data in tqdm(testloader):
            start_time = time.time()

            images, labels = data
            inputs, labels = images.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predict_end_time = time.time()

            elapsed_time = int((predict_end_time - start_time) * 1000)
            show_img(images[0], labels[0].item(), predicted[0].item(), elapsed_time)


if __name__ == '__main__':
    model_input_path = "./models/cifar10_model.pth"
    solver(model_input_path)
