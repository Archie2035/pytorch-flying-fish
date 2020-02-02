import torch.nn as nn
import torch.nn.functional as F
import torchvision


def get_flying_fish_net(resnet_name):
    backbone_dict = {
        "resnet18": torchvision.models.resnet18(pretrained=True),
        "resnet34": torchvision.models.resnet34(pretrained=True),
        "resnet50": torchvision.models.resnet50(pretrained=True),
        "resnet101": torchvision.models.resnet101(pretrained=True)
    }
    if resnet_name in backbone_dict:
        net = backbone_dict[resnet_name]
        net.conv1 = nn.Conv2d(3, 64, 5)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 10)
        return net
    else:
        raise Exception("not exists the net name")


def get_simple_net():
    net = SimpleNet()
    return net


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
