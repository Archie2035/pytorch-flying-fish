#!/usr/bin/env bash
python classifier_main.py --backbone resnet50 --epoch 20 --batch_size 16 --model_output_path ./models/cifar10_resnet50_model.pth