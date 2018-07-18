'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
    - train :
    - test :
'''
from __future__ import print_function

import pdb

import os
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

import numpy as np
import torchvision
import torchvision.transforms as transforms

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

if os.name == 'nt':
    term_width = int(4)
else:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

TOTAL_BAR_LENGTH = 45.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

## ______________________ Network Training________________________________
global best_acc  # best test accuracy
best_acc = 0
global train_acc
train_acc = 0
condenstation_mean = True

# Training
def train(data, model, epoch):
    global train_acc
    use_cuda  = model['use_cuda'] 
    net       = model['net']
    optimizer = model['optimizer']
    criterion = model['criterion']
    device    = model['device']

    trainloader = data['trainloader']

    if condenstation_mean:
        net.module.condenstation_mean()
        print("apply mean condenstation")

    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs, targets)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()        

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)#cross
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))
    train_acc = 100.*float(correct)/float(total)


def test(data, model, epoch):
    global best_acc
    use_cuda  = model['use_cuda'] 
    net       = model['net']
    optimizer = model['optimizer']
    criterion = model['criterion']
    device    = model['device']

    testloader = data['testloader']
    filename   = data['filename']

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

    # Save checkpoint.
    acc = 100.*float(correct)/float(total)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict() if use_cuda else net,
            'mean_vector' : net.module.mean_vector,
            'label_list' : net.module.label,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        filedir = './checkpoint/'
        state_file = filedir + filename+'_.t7'
        acc_file = filedir + filename+'_score.txt'
        torch.save(state, state_file)
        best_acc = acc
        File = open(acc_file,"w")
        File.write('Epoch: %d \n' % (epoch))
        File.write('Train Accuracy: %.3f %% \n' % (train_acc))
        File.write('Test Accuracy: %.3f %% \n' % (best_acc))
        File.close()

def mean_alignment(data, model, epoch):
    global best_acc
    use_cuda  = model['use_cuda'] 
    net       = model['net']
    optimizer = model['optimizer']
    criterion = model['criterion']
    device    = model['device']

    testloader = data['trainloader']
    filename   = data['filename']
    
    if condenstation_mean:
        net.module.condenstation_mean(flag=True)
        print("apply mean condenstation")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
                # outputs = net.get_feature(inputs)
            outputs = net(inputs, targets)            
            progress_bar(batch_idx, len(testloader))


def run(data, model, epoch, alignment = False):
    
    train(data,model,epoch)
    #torch.cuda.empty_cache()
    if alignment:

        print("Non-alignment")
        test(data,model,epoch)
        #torch.cuda.empty_cache()

        mean_alignment(data,model,epoch)
        #torch.cuda.empty_cache()

    model['scheduler'].step()
    test(data,model,epoch)
    #torch.cuda.empty_cache()

