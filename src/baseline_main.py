#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LRmodel


if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    # elif args.model == 'logistic':      # add for logistic regression
    #     global_model = LRmodel(784, 64)     # add for logistic regression
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    batch_size_constant = 64
    trainloader = DataLoader(train_dataset, batch_size=batch_size_constant, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            # images = torch.reshape(images, [-1, 1, 784])   # add for logistic regression
            optimizer.zero_grad()
            outputs = global_model(images)
            # print(outputs.shape)
            # print(labels)
            # outputs = torch.reshape(outputs, [-1, batch_size_constant, 1])   # add for logistic regression
            # loss = criterion(outputs, labels.view(len(labels), 1))  # add for logistic regression
            loss = criterion(outputs, labels) # origin code
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    print('Saving figure...')
    print(os.path.exists('D:/大三上学习/暑假大创/代码/晨霖联邦学习框架/save/'))
    plt.savefig('D:/大三上学习/暑假大创/代码/晨霖联邦学习框架/save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))
    print('Figure saved.')

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
