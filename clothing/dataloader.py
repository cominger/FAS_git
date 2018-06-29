import torch
import torch.utils.data as data
import os
from PIL import Image
import numpy as np

class Clothing1M(data.Dataset):
    """Clothing1M dataset
    input is image, target is annotation
    Arguments:
        root (string): filepath to rootder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root # /media/yonggis-yu/HDD2T1/dataset/clothing1M
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.labels = None
        # Annotations/%s.xml
        self._annopath = os.path.join(
            self.root, 'annotations', '%s')

        # Data/%s.jpg
        self._imgpath = os.path.join(
            self.root, 'Data', '%s')
        
        # "%s.txt" % "hi" => hi.txt
        with open(self._annopath % self.image_set) as f:
            result = np.loadtxt(f, dtype=np.unicode)            
            self.images = result[:, 0].tolist()
            self.labels = (result[:, 1].astype(int)).tolist()

    
    def __getitem__(self, index):
        img_name = self.images[index]
        target = self.labels[index]


        img = Image.open(self._imgpath % img_name).convert('RGB')
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    
    def __len__(self):
        return len(self.images)

    def class_num(self):
        label_list=[]
        for index in range(len(self.images)):
            label_list.append(self.labels[index])
        label_list = np.unique(np.array(label_list))
        return len(label_list)

    def append(self, other):
        self.images+=other.images
        self.labels+=other.labels
    
    def get_labeled_image_number(self, index_label):
        return (np.array(self.labels) == index_label).sum()

    def remove_from_list(self, index):
        self.images.pop(index)
        self.labels.pop(index)