import pdb
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from data_loader import numpy_2_dataset

import os
import time
import pickle
from PIL import Image

# from model import FASNET
# from model_margin_nmc_freez_mean import FASNET
# from model_mse_cross_entropy import FASNET
# from model_fc_with_margin_gt_vs_nmc import FASNET
from model_for_tsne import FASNET
from utils import utils

def show_images(epoch_n,class_n,images,RGB=True):
    output_dir="result";
    if not os.path.exists(output_dir):
    	os.mkdir(output_dir)

    epoch_dir=output_dir+"/"+str(epoch_n);
    if not os.path.exists(epoch_dir):
    	os.mkdir(epoch_dir)

    class_dir=epoch_dir+"/"+str(class_n);
    if not os.path.exists(class_dir):
    	os.mkdir(class_dir)

    for i, img in enumerate(images):
    	if RGB:
    		im = Image.fromarray(img)
    	else:
    		im = Image.fromarray(img[:,:,0])
    	png_name=class_dir+"/"+str(i)+'.png'
    	im.save(png_name)
        # plt.imshow(img)

def main():
	pdb.set_trace();

	# Hyper Parameters
	total_classes = 10
	num_classes = 10
	batch_size = 600

	# Initialize CNN
	# K = 53000 # total number of exemplars
	K = 500
	FAS = FASNET(batch_size, 2048, num_classes, True)
	epoch_n=0;
	s=0;
	FAS.cuda()

	# Load Datasets
	print ("Loading training examples for classes", range(s, s+num_classes));
	data_handler=utils("cifar10")
	# data_handler=utils("mix_mnist")
	train_set, test_set = data_handler.train_set, data_handler.test_set
	transform_input = data_handler.test_transform
	RGB=data_handler.RGB

	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
	                                           shuffle=True, num_workers=2)

	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
	                                           shuffle=True, num_workers=2)
	total = 0.0
	correct = 0.0
	nmc_correct = 0.0
	nmc_total = 0.0
	feature_concate=[]
	feature_fc_concate=[]
	label_concate=[]
	for indices, images, labels in train_loader:
	    images = Variable(images).cuda()

	    feature_origin=FAS.feature_extractor(images)
	    feature_map = FAS.forward(feature_origin)
	    labels = labels.long()              #addes for tensor type
	    if len(feature_concate):
	    	feature_concate=np.append(feature_concate,feature_origin.data.cpu().numpy(), axis=0)
	    	feature_fc_concate=np.append(feature_fc_concate,feature_map.data.cpu().numpy(),axis=0)
	    	label_concate=np.append(label_concate,labels.numpy(),axis=0)
	    else:
	    	feature_concate=feature_origin.data.cpu().numpy()
	    	feature_fc_concate=feature_map.data.cpu().numpy()
	    	label_concate=labels.numpy()

	np.savez('tsne_feature',feature_concate=feature_concate,feature_fc_concate=feature_fc_concate,label_concate=label_concate)
if __name__ == '__main__':
	main()
