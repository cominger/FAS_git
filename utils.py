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

TOTAL_BAR_LENGTH = 65.
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
condenstation_mean = False

# Training
def train(data, model, epoch):
    global train_acc
    use_cuda  = model['use_cuda'] 
    net       = model['net']
    optimizer = model['optimizer']
    criterion = model['criterion']

    trainloader = data['trainloader']

    if condenstation_mean:
        net.condenstation_mean()
        print("apply mean condenstation")

    print('\nEpoch: %d\n' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs, targets)

        targets_cpu = targets.data.cpu()
        targets_cpu = targets_cpu.map_(targets_cpu, lambda x,y: net.label.index(x))
        targets = Variable(targets_cpu).cuda()#non margin

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()        

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)#cross
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_acc = 100.*correct/total


def test(data, model, epoch):
    global best_acc
    use_cuda  = model['use_cuda'] 
    net       = model['net']
    optimizer = model['optimizer']
    criterion = model['criterion']

    testloader = data['testloader']
    filename   = data['filename']

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)


        targets_cpu = targets.data.cpu()
        targets_cpu = targets_cpu.map_(targets_cpu, lambda x,y: net.label.index(x))
        targets = Variable(targets_cpu).cuda()

        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.modules if use_cuda else net,
            'label_list' : net.label,
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
    else:
        print('Saving..')
        if not os.path.isdir('progress'):
            os.mkdir('progress')
        filedir = './progress/'
        acc_file = filedir + filename+'_score.txt'
        File = open(acc_file,"w")
        File.write('Epoch: %d \n' % (epoch))
        File.write('Train Accuracy: %.3f %% \n' % (train_acc))
        File.write('Test Accuracy: %.3f %% \n' % (acc))
        File.close()

def run(data, model, epoch):
    train(data,model, epoch)
    model['scheduler'].step()
    test(data,model, epoch)


def run_clothing(data,model,epoch):
    trainset  = data['trainset']
    class_num = len(set(trainset.labels))

    # #-----Learning Phase------
    # data['trainloader'] = torch.utils.data.DataLoader(trainset, batch_size=data['t_batch_size'], shuffle=True, num_workers=0)
    # train(data,model, epoch)
    # model['scheduler'].step()
    # test(data,model, epoch)

    #-----Cleaning Phase------
    if epoch % 1 ==0:
        print("Cleaning Time")
        sample_count = []
        for i in range(class_num):
            sample_count.append(trainset.get_labeled_image_number(i))
        sample_count=np.array(sample_count)

        data['trainloader'] = torch.utils.data.DataLoader(trainset, batch_size=data['t_batch_size'], shuffle=False, num_workers=0)
        noisy_sort(data,model,epoch,sample_count)
        trainset  = data['trainset']

        for mini_epoch in range(5):

            #-----Re Learning Phase------
            print("Big epoch: ", epoch)
            data['trainloader'] = torch.utils.data.DataLoader(trainset, batch_size=data['t_batch_size'], shuffle=True, num_workers=0)
            train(data,model, mini_epoch)
            model['scheduler'].step()
            test(data,model, mini_epoch)

        #load Best_model
        pre_trained_path = './checkpoint/Clothing1M_ncm_deep_rest50_.t7'
        checkpoint                = torch.load(pre_trained_path)
        pre_trained_model         = checkpoint['net'].__self__
        model['net'].load_state_dict(pre_trained_model.state_dict())
        model['net'].mean_vector  = checkpoint['mean_vector']
        model['net'].label        = checkpoint['label_list']
        model['net'].count_vector = torch.ones(len(checkpoint['label_list']),1)


def noisy_sort(data,model,epoch,sample_count):
    global best_acc
    if best_acc == 0:
        best_acc = 50.0
    remove_percent = (100 - best_acc)/100
    sample_count = sample_count * remove_percent

    use_cuda  = model['use_cuda'] 
    net       = model['net']
    optimizer = model['optimizer']
    criterion = model['criterion']

    trainloader = data['trainloader']
    filename   = data['filename']
    trainset = data['trainset']

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    remove_list_score=[]
    remove_list_index=[]

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(batch_idx)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        targets_cpu = targets.data.cpu()
        targets_cpu = targets_cpu.map_(targets_cpu, lambda x,y: net.label.index(x))
        targets = Variable(targets_cpu).cuda()

        progress_bar(batch_idx, len(trainloader))
        # total += targets.size(0)

        mean_Dist = outputs.index_select(1,targets).diag().data.cpu().numpy()
        del targets
        targets_cpu  = targets_cpu.numpy()
        batch_idx_in = np.arange(inputs.shape[0])+(batch_idx*data['t_batch_size'])

        for cur_label in range(len(set(trainset.labels))):
            cur_index     = (targets_cpu == cur_label)
            cur_Dist      = mean_Dist[cur_index]
            cur_batch_idx =  batch_idx_in[cur_index]
            if batch_idx == 0:
                remove_list_score.append(cur_Dist)
                remove_list_index.append(cur_batch_idx)
            else:
                remove_list_score[cur_label]=np.hstack((remove_list_score[cur_label],cur_Dist))
                remove_list_index[cur_label]=np.hstack((remove_list_index[cur_label],cur_batch_idx))

        for cur_label in range(len(set(trainset.labels))):
            sort_list = np.argsort(remove_list_score[cur_label])
            remove_list_score[cur_label] = remove_list_score[cur_label][sort_list]
            remove_list_index[cur_label] = remove_list_index[cur_label][sort_list]
            remove_list_score[cur_label] = remove_list_score[cur_label][:int(sample_count[cur_label])]
            remove_list_index[cur_label] = remove_list_index[cur_label][:int(sample_count[cur_label])]

    del remove_list_score
    remove_list_index = [y for x in remove_list_index for y in x]
    remove_list_index.sort(reverse=True)
    for  cur_label in remove_list_index:
        trainset.remove_from_list(cur_label)


