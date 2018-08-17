"""
'''Feature histogram of LSUN'''
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
from PIL import Image
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from models.resnet_imagenet_ncm import *
from utils_ncm import *

import clothing.dataloader as cd

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# Data
thread_workers = 8
t_batch_size = 10 #limit 100



def feature_save(dataloader, net, b_idx):
    
  
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        print(batch_idx)
        if batch_idx == b_idx: break;
        
        if torch.cuda.is_available:
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        outputs = net.get_feature(inputs)
        outputs = outputs.data.cpu()
        targets = targets.cpu()
        inputs = inputs.cpu()
        if batch_idx == 0:
            data = outputs
            labels = targets
            data_pix = inputs
        else:
            data = torch.cat((data, outputs))
            labels = torch.cat((labels, targets))
            data_pix = torch.cat((data_pix, inputs))

    return data, labels, data_pix
     
def feature_output(dataloader, net, max_len, nlabels):

    print('sample count')
    net.eval()
    num_class = nlabels
    total_length=[]
    wrong_length=[]

    correct = 0
    total = 0   
 
    with torch.no_grad(): 
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            print("progress: %5.2f" %(batch_idx/float(max_len)*100*100),end='\r')

            predict, feature = net(inputs)
            _, predict = torch.max(predict.data,1)

	        total += targets.size(0)
            correct += predict.eq(targets.data).cpu().sum()

            #progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)'
            #    % (100.*float(correct)/float(total), correct, total))
            print(100.*float(correct)/float(total), correct, total) 


            outputs = feature
            outputs = outputs.pow(2)
            outputs = torch.sum(outputs,dim=1)

            predict = predict.data.cpu().numpy()
            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()

            #GT count
            for cur_class in range(num_class):
                # print (cur_class)
                idx       = [targets==cur_class]
                idx_wrong = [x&y for (x,y) in zip([predict != cur_class],idx)]
                if batch_idx ==0:
                    total_length.append(outputs[idx])
                    wrong_length.append(outputs[idx_wrong])
                else:
                    total_length[cur_class] = np.append(total_length[cur_class],outputs[idx])
                    wrong_length[cur_class] = np.append(wrong_length[cur_class],outputs[idx_wrong])

    print('\n histogram')
    for cur_class in range(num_class):
        # print(total_lenght_list[cur_class])
        cur_value = net.module.mean_vector[cur_class]
        cur_value = torch.sum(cur_value.pow(2))
        cur_value = cur_value.data.cpu().numpy()

        print ("mean vector %5.2f" %(cur_value),end='\r')
        hist,bins     = np.histogram(total_length[cur_class], bins = 100)
        w_hist,w_bins = np.histogram(wrong_length[cur_class], bins = bins)
        width = 0.7 * (bins[1]-bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        # plt.subplot(1,2,1)
        plt.bar(center,hist,   align = 'center', width= width)
        plt.bar(center,w_hist, align = 'center', width= width, alpha=0.5,color ='red')
        title_name = 'class: '+ str(cur_class)+ ' total sample histogram ' + str(cur_value)
        plt.title(title_name)
        plt.ylabel('count')

        # plt.subplot(1,2,2)
        # plt.bar(center,w_hist,align = 'center', width= width)
        # title_name = 'mean: '+ str(cur_value)+ 'incorrect sample histogram'
        # plt.title(title_name)
        # plt.ylabel('count')

        # plt.show()
        file_name='./progress_plot/class_'+str(cur_class)+'.png'
        plt.savefig(file_name)
        plt.clf()
        # print ()



def main():
    import pdb
    pdb.set_trace()

    # Data
    #thread_workers = 24
    #t_batch_size = 256 #limit 100

    #prepare data
    root= '/media/cvpr-mu/4TB_1/ckmoon/dataset/lsun'
    lsun_categories = ['bedroom_train', 'bridge_train', 'church_outdoor_train', 'classroom_train', 'conference_room_train', 'dining_room_train', 'kitchen_train', 'living_room_train', 'restaurant_train']#, 'tower_train']

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
    ])

    print('==> Preparing data..')
    trainset = datasets.LSUN(root,lsun_categories, transform_test)
    nlabels = len(trainset.classes)
    trainloader = data.DataLoader(trainset, batch_size = t_batch_size, shuffle = False ,num_workers = thread_workers)    

    lsun_categories = ['bedroom_val', 'bridge_val', 'church_outdoor_val', 'classroom_val', 'conference_room_val', 'dining_room_val', 'kitchen_val', 'living_room_val', 'restaurant_val']#, 'tower_train']
    testset = datasets.LSUN(root, lsun_categories, transform_test)
    nlabels = len(testset.classes)
    testloader = data.DataLoader(testset, batch_size = t_batch_size,shuffle = False ,num_workers = thread_workers)

    pre_trained_path = './checkpoint/LSUN_deep_rest50_pretrained_imagenet_noisy_dataset_true_U_sgd_1_.t7'
    pre_load = torch.load(pre_trained_path)
    pre_trained_model = pre_load['net']
    
    net = resnet34_dnc_imagenet(num_classes=nlabels)

    if use_cuda:
        net = torch.nn.DataParallel(net) 
        cudnn.benchmark = True

    net.load_state_dict(pre_trained_model)
    net = net.to(device)

    #feature_output(trainloader, net, len(trainset), nlabels)
    feature_output(testloader, net, len(testset), nlabels)

main()
