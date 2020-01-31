# pytorch_quick_start
作为一个新手，想开始进入深度学习领域做一名炼丹师。通过一个最简的原型来学习一下pytorch的玩法。

## 0.准备工作
使用conda创建虚拟环境
```
conda create -n pytorch-qs python=3.7 -y
conda activate pytorch-qs
```

安装PyTorch and torchvision，参照[官网](https://pytorch.org/),例如：
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

克隆本项目
```
git clone https://github.com/tianzhihen/pytorch_quick_start.git
cd pytorch_quick_start
```

安装依赖包
```
pip install -r requirements.txt
```

training
```
python classifier_pipeline.py
```

## 1.架构

1. [x] 实现简单的Neural Network、支持training、testing
1. [x] 在GPU下训练
1. [x] 接入tensor_board
1. [ ] 简易的benchmark
1. [ ] 使用hook重构
1. [ ] 适配多个数据集
1. [ ] dataloader 异步加速
1. [ ] 加入resnet 做为backbone
1. [ ] 抽象配置文件

## 2.实现简单的Neural Network
通过pytorch的60min教程搭建一个包含conv层maxpooling层以及relu激活函数的nn，官网[链接](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py)

neural network 各层的输入输出关系，可以通过LeNet的说明进行理解。
![](doc/LeNet.png)
![](doc/LeNet_info.png)

## 3. GPU训练
将net放到gpu上：
```
net.to(device)
```
将输入数据放入到gpu上：
```
inputs, labels = inputs.to(device), labels.to(device)
```


## 4.接入tensor_board
观察图像：
```
writer.add_image('train_images', img)
```
观察网络结构：
```
writer.add_graph(net, images)
```

观察loss：
```
writer.add_scalar('training loss', loss, index * len(trainloader) + i)
```

开启tensorboard：
```
tensorboard --logdir=runs --host=0.0.0.0
```

图像示例：
![](doc/tensorboard_image.png)

网络结构示例：
![](doc/tensorboard_graph.png)

loss示例：
![](doc/tensorboard_loss.png)


 