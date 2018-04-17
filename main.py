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
# from model_fc_with_margin import FASNET
from model_margin_nmc import FASNET
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
	# pdb.set_trace();

	# Hyper Parameters
	total_classes = 10
	num_classes = 10
	batch_size = 600

	# Initialize CNN
	# K = 53000 # total number of exemplars
	K = 500
	FAS = FASNET(batch_size, 2048, num_classes)
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

	for epoch_n in range(30):

	    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
	                                               shuffle=True, num_workers=2)

	    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
	                                               shuffle=True, num_workers=2)

	    # Update representation via BackProp
	    FAS.update_representation(train_set,transform_input)
	    # if epoch_n==0:
	    	# FAS.update_representation(train_set,transform_input,True)
	    # else:
	    	# FAS.update_representation(train_set,transform_input,False)
	    m = int(K / FAS.n_classes)

	    # Construct exemplar sets for new classes
	    print("current exemplar number : %d" %(len(FAS.exemplar_sets)))
	    FAS.exemplar_sets=[];
	    for y in range(FAS.n_known, FAS.n_classes):
	        print ("Constructing exemplar set for class-%d..." %(y))
	        #Batch
	        images = train_set.get_image_class(y)
	        images_label = np.ones((images.shape[0])) * y
	        dl_images = numpy_2_dataset(images, images_label, transform_input)
	        FAS.construct_exemplar_set(dl_images, images, m, transform_input)
	        print ("Done")

	    for y, P_y in enumerate(FAS.exemplar_sets):
	        print ("Exemplar set for class-%d:" % (y), P_y.shape)
	        show_images(epoch_n,y,P_y,RGB) # RGB

	    #Save and load model
	    save_model_path="result"+"/"+str(epoch_n)+"/model.pkl";
	    torch.save(FAS.state_dict(),save_model_path)
	    save_exemplar_path="result"+"/"+str(epoch_n)+"/exemplar.pkl";
	    with open(save_exemplar_path,"wb") as file:
	    	pickle.dump(FAS.n_classes, file)
	    	pickle.dump(FAS.exemplar_sets, file)
	    	pickle.dump(FAS.exemplar_means, file)
	    save_model_accracy="result"+"/"+str(epoch_n)+"/score.txt";
	    file_score = open(save_model_accracy,"w")
	    # save_model=copy.deepcopy(FAS);
	    # with open(save_path,"wb") as file:
	    # 	pickle.dump(save_model, file)

	    # FAS.load_state_dict(torch.load(save_path))
	    # FAS.n_known = FAS.n_classes
	    # print ("FAS classes: %d" % FAS.n_known)
		# with open(save_path,'rb') as file:
	 #    	adef = pickle.load(save_model, file)
	    total = 0.0
	    correct = 0.0
	    nmc_correct = 0.0
	    nmc_total = 0.0
	    for indices, images, labels in train_loader:
	        images = Variable(images).cuda()
	        preds = FAS.classify(images, transform_input)
	        labels = labels.long()              #addes for tensor type
	        total += labels.size(0)
	        correct += (preds.data.cpu() == labels).sum()

	        # preds = FAS.soft_classify(images, transform_input)
	        # nmc_total += labels.size(0)
	        # nmc_correct += (preds.data.cpu() == labels).sum()

	    print('Sample Train Accuracy: %d %%' % (100 * correct / total))
	    # print('soft Train Accuracy: %d %%' % (100 * nmc_correct / nmc_total))
	    file_score.write('Sample Train Accuracy: %d %% \n' % (100 * correct / total))
	    # file_score.write('soft_Train Accuracy: %d %% \n' % (100 * nmc_correct / nmc_total))

	    total = 0.0
	    correct = 0.0
	    nmc_correct = 0.0
	    nmc_total = 0.0
	    for indices, images, labels in test_loader:
	        images = Variable(images).cuda()
	        preds = FAS.classify(images, transform_input)
	        labels = labels.long()              #addes for tensor type
	        total += labels.size(0)
	        correct += (preds.data.cpu() == labels).sum()

	        # preds = FAS.soft_classify(images, transform_input)
	        # nmc_total += labels.size(0)
	        # nmc_correct += (preds.data.cpu() == labels).sum()

	    print('Sample Test Accuracy: %d %%' % (100 * correct / total))	    
	    # print('soft Test Accuracy: %d %%' % (100 * nmc_correct / nmc_total))	    

	    file_score.write('Sample Test Accuracy: %d %% \n' % (100 * correct / total))
	    # file_score.write('soft_Test Accuracy: %d %% \n' % (100 * nmc_correct / nmc_total))
	    file_score.close()

if __name__ == '__main__':
	main()
