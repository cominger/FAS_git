import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# from models.dnc_imagenet_FAS import *
from models.resnet_imagenet_ncm import *
import utils 
from torch.autograd import Variable

import clothing.dataloader as cd



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
     
def feature_output(dataloader,net, max_len):
    print('sample count')
    num_class=14
    # total_length=np.zeros((num_class,1))
    total_length=[]
    wrong_length=[]
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print("progress: %5.2f" %(batch_idx/float(max_len)*100*100),end='\r')
        if torch.cuda.is_available:
            inputs = inputs.cuda()
        inputs  = Variable(inputs)

        predict = net(inputs)
        _, predict = torch.max(predict.data,1)

        outputs = net.get_feature(inputs)
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

    print('histogram')
    for cur_class in range(num_class):
        # print(total_lenght_list[cur_class])
        cur_value = net.mean_vector[cur_class]
        cur_value = torch.sum(cur_value.pow(2))
        cur_value = cur_value.data.cpu().numpy()

        print ("mean vector %5.2f" %(cur_value),end='\r')
        hist,bins     = np.histogram(total_length[cur_class], bins = 100)
        w_hist,w_bins = np.histogram(wrong_length[cur_class], bins = bins)
        width = 0.7 * (bins[1]-bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        # plt.subplot(1,2,1)
        plt.bar(center,hist,   align = 'center', width= width)
        plt.bar(center,w_hist, align = 'center', width= width, alpha=0.5)
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

    pre_trained_path = './checkpoint/Clothing1M_ncm_deep_rest50_clean_dataset_.t7'
    #result_path = pre_trained_path[:-3]+"_feature.pkl"

    # Data
    t_batch_size=100 #limit 100

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

    clothing1m_clean_train = cd.Clothing1M(root, 'clean_train_kv.txt', transform=transform_test)
    clothing1m_noise_train = cd.Clothing1M(root, 'noisy_train_kv.txt', transform=transform_test)
    # trainset = clothing1m_noise_train
    # trainset = torch.utils.data.ConcatDataset((clothing1m_clean_train, clothing1m_noise_train))
    clean_trainloader = torch.utils.data.DataLoader(clothing1m_clean_train, batch_size=t_batch_size, shuffle=True, num_workers=0)
    noisy_trainloader = torch.utils.data.DataLoader(clothing1m_noise_train, batch_size=t_batch_size, shuffle=True, num_workers=0)

    clothing1m_clean_test = cd.Clothing1M(root, 'clean_test_kv.txt', transform=transform_test)
    testloader = torch.utils.data.DataLoader(clothing1m_clean_test, batch_size=t_batch_size, shuffle=False, num_workers=0)

 
    load = torch.load(pre_trained_path)
    pre_trained_model = load['net'].__self__
    
    net = resnet50_dnc_imagenet()
    net.load_state_dict(pre_trained_model.state_dict())
    net.mean_vector = load['mean_vector']
    net.label_list = load['label_list']
    if torch.cuda.is_available:
        net = net.cuda()

    net.eval()
    class_test = 0
    # pdb.set_trace()

    # feature_output(clean_trainloader,net)
    # feature_output(testloader, net, len(clothing1m_clean_test))
    feature_output(clean_trainloader, net, len(clothing1m_clean_train))
    # feature_output(noisy_trainloader, net, len(clothing1m_noise_train))

    # clean_data, clean_labels, clean_pix = feature_save(clean_trainloader, net, 1)
    # clean_data_numpy   = clean_data.cpu().numpy()
    # clean_labels_numpy = clean_labels.cpu().numpy()
    # clean_pix_numpy    = clean_pix.cpu().data.numpy()
    # del clean_data, clean_labels, clean_pix

    # noisy_data, noisy_labels, noisy_pix = feature_save(noisy_trainloader, net, 50)
    # noisy_data_numpy   = noisy_data.cpu().numpy()
    # noisy_labels_numpy = noisy_labels.cpu().numpy()
    # noisy_pix_numpy    = noisy_pix.cpu().data.numpy()
    # del noisy_data, noisy_labels, noisy_pix

    # clean_zero     = clean_data_numpy[clean_labels_numpy==class_test]
    # clean_zero_pix = clean_pix_numpy[clean_labels_numpy==class_test]
    # clean_zero_label = clean_labels_numpy[clean_labels_numpy==class_test] 

    # noisy_zero     = noisy_data_numpy[noisy_labels_numpy==class_test]
    # noisy_zero_pix = noisy_pix_numpy[noisy_labels_numpy==class_test]
    # noisy_zero_label = noisy_labels_numpy[noisy_labels_numpy==class_test] 

    # # num_size=100
    # # clean_zero       = clean_data_numpy[:num_size]
    # # clean_zero_pix   = clean_pix_numpy[:num_size]
    # # clean_zero_label = clean_labels_numpy[:num_size]

    # # noisy_zero       = noisy_data_numpy[:num_size]
    # # noisy_zero_pix   = noisy_pix_numpy[:num_size]
    # # noisy_zero_label = noisy_labels_numpy[:num_size] 
    
    # class_zero_mean = net.mean_vector[class_test].cpu().numpy()

    # clean_zero = np.power(clean_zero-class_zero_mean,2)
    # clean_zero = np.sum(clean_zero , axis=1)
    # noisy_zero = np.power(noisy_zero-class_zero_mean,2)
    # noisy_zero = np.sum(noisy_zero , axis=1)

    # clean_zero_index = np.argsort(clean_zero)
    # noisy_zero_index = np.argsort(noisy_zero)
    # total_sum = np.hstack((clean_zero,noisy_zero))
    # print(np.sort(total_sum))

    # t_length   = (clean_zero_index.shape[0]+noisy_zero_index.shape[0])
    # clean_zero  = clean_zero[clean_zero_index]
    # clean_pix   = clean_zero_pix[clean_zero_index]
    # clean_label = clean_zero_label[clean_zero_index]

    # noisy_zero  = noisy_zero[noisy_zero_index]
    # noisy_pix   = noisy_zero_pix[noisy_zero_index]
    # noisy_label = noisy_zero_label[noisy_zero_index]
    
    # cur_clean=0;
    # cur_noisy=0;
    # total      = [];
    # total_pix  = [];
    # total_from = [];
    # for i in range(t_length):
    #     # print("(%d , %d)\n" %(cur_clean,cur_noisy))
    #     if cur_clean == clean_zero_index.shape[0]:
    #         total.append(noisy_zero[cur_noisy])
    #         total_pix.append(noisy_pix[cur_noisy])
    #         name = 'noisy_'+str(noisy_label[cur_noisy])
    #         total_from.append(name)
    #         cur_noisy+=1;
    #         continue;

    #     if cur_noisy == noisy_zero_index.shape[0]:
    #         total.append(clean_zero[cur_clean])
    #         total_pix.append(clean_pix[cur_clean])
    #         name = 'clean_'+str(clean_label[cur_clean])
    #         total_from.append(name)
    #         cur_clean+=1;
    #         continue;

    #     if clean_zero[cur_clean] >= noisy_zero[cur_noisy]:
    #         total.append(noisy_zero[cur_noisy])
    #         total_pix.append(noisy_pix[cur_noisy])
    #         name = 'noisy_'+str(noisy_label[cur_noisy])
    #         total_from.append(name)
    #         cur_noisy+=1;

    #     else:
    #         total.append(clean_zero[cur_clean])
    #         total_pix.append(clean_pix[cur_clean])
    #         name = 'clean_'+str(clean_label[cur_clean])
    #         total_from.append(name)
    #         cur_clean+=1;
    # print(total_from)

    # output_dir="result";
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # for i in range(t_length):
    #     png_name = output_dir+"/"+str(i)+"_"+total_from[i]+"_score_"+str(total[i])+'.png'
    #     cur_pix=(np.swapaxes(total_pix[i],0,-1))
    #     cur_pix=(np.swapaxes(cur_pix,0,1))
    #     im = Image.fromarray((cur_pix*255).astype(np.uint8))
    #     im.save(png_name)
    # # result['data'] = data.numpy()
    # # result['labels'] = labels.numpy().tolist()

    # # with open(result_path, 'wb') as f:
    # #     pkl.dump(result, f)

main()
