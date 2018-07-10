'''Train Clothing1M with PyTorch.'''
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

from models.resnet_imagenet_ncm import *
# from models.resnet_imagenet import *
from utils_ncm import *
# from utils import *
from torch.autograd import Variable

import clothing.dataloader as cd

import pdb

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--debug', '-d', action ='store_true', help ='enable pdb')
parser.add_argument('--batch_size', '-bs', default=40, help='batch_size')
args = parser.parse_args()
# filename = 'Clothing1M_deep_rest50_noise_dataset_with_alignment_imagenet_pretrained_sgd'
filename = 'Clothing1M_deep_rest50_clean_dataset_with_alignment_imagenet_pretrained_sgd'

def main():
    if args.debug:
        pdb.set_trace()
    model={}
    data_set={}
    use_cuda = torch.cuda.is_available()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    end_epoch = 300
    lr_step = [100, 150, 200, 250]
    t_batch_size = args.batch_size

    device = torch.device("cuda" if use_cuda else "cpu")

    # pdb.set_trace()
    #prepare data
    root= './data/Clothing1M/'
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    print('==> Preparing data..')
    clothing1m_clean_train = cd.Clothing1M(root, 'clean_train_kv.txt', transform=transform_train)
    clothing1m_noise_train = cd.Clothing1M(root, 'noisy_train_kv.txt', transform=transform_train)
    # clothing1m_clean_train.append(clothing1m_noise_train)
    trainset = clothing1m_clean_train
    # trainset = clothing1m_noise_train
    # trainset = torch.utils.data.ConcatDataset((clothing1m_clean_train, clothing1m_noise_train))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=t_batch_size, shuffle=True, num_workers=16)

    clothing1m_clean_test = cd.Clothing1M(root, 'clean_test_kv.txt', transform=transform_test)
    testset = clothing1m_clean_test
    testloader = torch.utils.data.DataLoader(testset, batch_size=int(t_batch_size), shuffle=False, num_workers=16)


    # Model
    global net
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        net = resnet50_dnc_imagenet(pretrained= True)

        pre_trained_path = './checkpoint/Clothing1M_ncm_deep_rest50_dirty_.t7'
        checkpoint = torch.load(pre_trained_path)
        pre_trained_model = checkpoint['net'].__self__
        net.load_state_dict(pre_trained_model.state_dict())
        net.mean_vector  = checkpoint['mean_vector']
        net.label        = checkpoint['label_list']
        net.count_vector = torch.ones(len(checkpoint['label_list']),1)

    else:
        print('==> Building model..')
        # net = VGG('VGG19')
        net = resnet50_dnc_imagenet(pretrained= True, num_classes=trainset.class_num()) #NMC
        # net = resnet50(pretrained= True)
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
        net = torch.nn.DataParallel(net,device_ids=[0,1])
     #    net.module.mean_vector  = net.module.mean_vector.to(device)
    	# net.module.count_vector = net.module.count_vector.to(device)
        cudnn.benchmark  = True

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MultiLabelMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)
    

    # pdb.set_trace()
    model['use_cuda']  = use_cuda
    model['net']       = net
    model['criterion'] = criterion
    model['optimizer'] = optimizer
    model['scheduler'] = scheduler
    model['device']    = device

    for epoch in range(start_epoch, end_epoch):

        data_set['trainset']    = trainset
        data_set['trainloader'] = trainloader
        data_set['testset']     = testset
        data_set['testloader']  = testloader
        data_set['filename']    = filename
        
	run(data_set,model, epoch)


if __name__ == '__main__':
    main()
