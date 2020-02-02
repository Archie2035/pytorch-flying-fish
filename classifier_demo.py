import torch
from networks.flying_fish_net import get_flying_fish_net, get_simple_net
import torchvision
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Classifier')
    parser.add_argument('--batch_size', type=int, default=4, help='mini_batch_size')
    parser.add_argument('--backbone', default='', help='backbone, e.g. resnet18、resnet50、resnet101...')
    parser.add_argument('--model_input_path', default='./models/cifar10_model.pth', help='input model path')

    args = parser.parse_args()
    return args


def show_img(img, label, predict, elapsed_time):
    img = img * 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.title("label:{}\npredict:{}\nelapsed_time:{}ms".format([classes[i] for i in label],
                                                               [classes[i] for i in predict], elapsed_time))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def solver(args):
    if args.backbone:
        net = get_flying_fish_net(args.backbone)
    else:
        net = get_simple_net()

    net.load_state_dict(torch.load(args.model_input_path))
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
            show_img(torchvision.utils.make_grid(images), labels.tolist(), predicted.tolist(), elapsed_time)


if __name__ == '__main__':
    args = parse_args()

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    solver(args)
