'''Train UCF101 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.dnc_imagenet_FAS import * ## NCM ResNET
#from torchvision.models import *       ## original

from utils_ncm  import *
from torch.autograd import Variable

import UCF101.dataloader 

import pdb

parser = argparse.ArgumentParser(description='PyTorch UCF101 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--debug', '-d', action='store_true', help = 'pdb enable')
args = parser.parse_args()
filename = 'UCF101_ncm_deep_rest152_adam'

def main():
    if args.debug:
        pdb.set_trace()

    model={}
    data_set={}
    use_cuda = torch.cuda.is_available()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    end_epoch = 300
    lr_step = [150, 250]

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
	transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    t_batch_size=50

    trainset = UCF101.dataloader.UCF101(root='./data',image_set='./data/TH14/th_train_path2Class.txt', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=t_batch_size, shuffle=True, num_workers=5)

    testset = UCF101.dataloader.UCF101(root='./data', image_set='./data/TH14/th_test_path2Class.txt' ,transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=t_batch_size, shuffle=False, num_workers=5)

    data_set['trainset']    = trainset
    data_set['trainloader'] = trainloader
    data_set['testset']     = testset
    data_set['testloader']  = testloader
    data_set['filename']    = filename

    # Model
    global net
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        # net = VGG('VGG19')
        net = resnet101_dnc_imagenet(pretrained=True)   #NCM
        #net = resnet152(pretrained=True)
        # net = ResNet18_DNC()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()

    if use_cuda:
        net.cuda()
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MultiLabelMarginLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)
    

    # pdb.set_trace()
    model['use_cuda']  = use_cuda
    model['net']       = net
    model['criterion'] = criterion
    model['optimizer'] = optimizer
    model['scheduler'] = scheduler

    for epoch in range(start_epoch, end_epoch):
        # pdb.set_trace()
        # test(epoch)
        # train(model, epoch)
        # scheduler.step()
        # test(net, epoch)
        run(data_set,model, epoch)
    
if __name__ == '__main__':
    main()
