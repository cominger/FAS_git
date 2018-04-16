from torchvision.datasets import CIFAR10, MNIST, folder
from torch.utils.data import Dataset
import numpy as np
import os
import scipy.io
import torch
import math
from PIL import Image

class numpy_2_dataset(Dataset):
    def __init__(self, data_tensor, target_tensor, transform=None):
        self.train_data = data_tensor
        self.train_labels = target_tensor 
        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self,index):
        img, target = self.train_data[index], self.train_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return index, img, target

class ori_CIFAR10(CIFAR10):
    def __init__(self, root,
                 classes=range(10),
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(ori_CIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.train_data)):
                if self.train_labels[i] in classes:
                    train_data.append(self.train_data[i])
                    train_labels.append(self.train_labels[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    test_data.append(self.test_data[i])
                    test_labels.append(self.test_labels[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels

class ori_CIFAR100(ori_CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    def __init__(self, root,
                     classes=range(100),
                     train=True,
                     transform=None,
                     target_transform=None,
                     download=False):
        super(ori_CIFAR100,self).__init__(root= root,
                                         classes=classes,
                                         train=train,
                                         transform=transform,
                                         target_transform=target_transform,
                                         download=download)

class ori_MNIST(MNIST):
    def __init__(self, root,
             classes=range(10),
             train=True,
             transform=None,
             target_transform=None,
             download=False):
        super(ori_MNIST, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.train_data)):
                    train_data.append(self.train_data[i].numpy())
                    train_labels.append(self.train_labels[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.test_data)):
                    test_data.append(self.test_data[i].numpy())
                    test_labels.append(self.test_labels[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels

class noi_MNIST(object):
    def __init__(self, root,
             classes=range(10),
             train=True,
             transform=None,
             target_transform=None,
             download=False):

        # Select subset of classes
        if train:
            self.train_data = scipy.io.loadmat(os.path.join(root,'noi_raw','train_x.mat'))["train_x"]
            sample_num=self.train_data.shape[0];
            sample_rows=sample_cols=int(math.sqrt(self.train_data.shape[1]));
            self.train_data = self.train_data.reshape(sample_num,sample_rows,sample_cols)
            self.train_labels = scipy.io.loadmat(os.path.join(root,'noi_raw','train_y.mat'))["train_y"]
            labels=[];
            for one_hot in self.train_labels:
                labels.append(np.nonzero(one_hot)[0][0])
            self.train_labels=labels

        else:
            self.test_data = scipy.io.loadmat(os.path.join(root,'noi_raw','test_x.mat'))["test_x"]
            sample_num=self.test_data.shape[0];
            sample_rows=sample_cols=int(math.sqrt(self.test_data.shape[1]));
            self.train_data = self.train_data.reshape(sample_num,sample_rows,sample_cols)
            self.test_labels = scipy.io.loadmat(os.path.join(root,'noi_raw','test_y.mat'))["test_y"]

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels

class combined_MNIST(MNIST):
    def __init__(self, root,
             classes=range(10),
             train=True,
             transform=None,
             target_transform=None,
             download=False,
             mnistdata=None,
             nmnistdata=None,
             mix=0.3):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train=train
        if self.train:

            input_mnist=np.stack((mnistdata.train_data, mnistdata.train_data, mnistdata.train_data),axis=-1)
            self.train_data=input_mnist
            self.train_labels=mnistdata.train_labels

            nmnistdata_proceed=[];
            nmnistlabels_proceed=[];
            for class_num in classes:
                nmnistdata_raw=nmnistdata.get_image_class(class_num);
                count=int(len(nmnistdata_raw)*mix);
                print("mixing sample: ",count)
                for c in range(count):
                    rand_index=c;
                    nmnistdata_proceed.append(nmnistdata_raw[rand_index]);
                    nmnistlabels_proceed.append(class_num);

            if(nmnistdata_proceed != []):
                input_nmist=np.stack((nmnistdata_proceed, nmnistdata_proceed, nmnistdata_proceed),axis=-1)
                self.train_data=np.concatenate((input_mnist, input_nmist),axis=0)
                self.train_labels=np.concatenate((mnistdata.train_labels, nmnistlabels_proceed), axis=0)

        else:
            input_mnist=np.stack((mnistdata.test_data, mnistdata.test_data, mnistdata.test_data),axis=-1)
            self.test_data=input_mnist
            self.test_labels=mnistdata.test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = np.concatenate((self.train_labels,labels), axis=0)

class cifar_ImageFolder(folder.ImageFolder):
    def __init__(self, root,
             transform=None, 
             target_transform=None):
        super(cifar_ImageFolder, self).__init__(root,
                                               transform=transform,
                                               target_transform=target_transform)
        perfer_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
        perfer_class_to_idx = {perfer_classes[i]: i for i in range(len(self.classes))}

        self.train_data = [];
        self.train_labels = [];
        for path, target in self.imgs:
            cur_image=self.loader(path)
            cur_image=np.array(cur_image.resize((32,32)))
            if cur_image.shape[2] != 3:
                continue;
            self.train_data.append(cur_image)
            ori_class = self.classes[target]
            self.train_labels.append(perfer_class_to_idx[ori_class])
        self.train_data=np.array(self.train_data)
        self.train_labels=np.array(self.train_labels)
        self.calsses=perfer_classes
        self.class_to_idx = perfer_class_to_idx 

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = np.concatenate((self.train_labels,labels), axis=0)

class combine_cifar(CIFAR10):
    def __init__(self, root,
             classes=range(10),
             train=True,
             transform=None,
             target_transform=None,
             ori_cifar=None,
             web_cifar=None,
             download=False):
            super(combine_cifar, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
            if self.train:

                self.train_data=np.concatenate((ori_cifar.train_data,web_cifar.train_data),axis=0)
                self.train_labels=np.concatenate((ori_cifar.train_labels,web_cifar.train_labels),axis=0)

                # self.train_data = np.array(train_data)
                # self.train_labels = train_labels

            else:
                self.test_data=ori_cifar.test_data
                self.test_labels=ori_cifar.test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels