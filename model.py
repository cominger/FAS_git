import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
from PIL import Image

from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from data_loader import numpy_2_dataset

# Hyper Parameters
num_epochs = 3
batch_size = 100
learning_rate = 0.002

class FASNET(nn.Module):
    def __init__(self, feature_size, n_classes):
        # Network architecture
        super(FASNET, self).__init__()
        self.feature_extractor = resnet18()
        self.feature_extractor.fc =\
            nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_classes, bias=False)

        self.n_classes = n_classes
        self.n_known = 0
        self.batch_size=batch_size

        # List containing exemplar_sets
        # Each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        self.exemplar_sets = []

        # Learning method
        self.cls_loss = nn.CrossEntropyLoss()
        self.cms_loss = nn.MarginRankingLoss()
        # self.dist_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        #self.optimizer = optim.SGD(self.parameters(), lr=2.0,
        #                           weight_decay=0.00001)

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features+n, bias=False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def soft_classify(self, x, transform):

        batch_size = x.size(0)

        feature = self.forward(x) # (batch_size, feature_size)
        _, preds = torch.max(feature,1)

        return preds

    def classify(self, x, transform):
        """Classify images by neares-means-of-exemplars

        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        batch_size = x.size(0)

        if self.compute_means:
            print ("Computing mean of exemplars...")
            exemplar_means = []
            for P_y in self.exemplar_sets:
                features = []
                # Extract feature for each exemplar in P_y
                for ex in P_y:
                    ex = Variable(transform(Image.fromarray(ex)), volatile=True).cuda()
                    feature = self.feature_extractor(ex.unsqueeze(0))
                    # class_f = self.forward(ex)
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm() # Normalize
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            print ("Done")

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means) # (n_classes, feature_size)
        means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

        feature = self.feature_extractor(x) # (batch_size, feature_size)
        for i in range(feature.size(0)): # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
        feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
        _, preds = dists.min(1)

        return preds
        

    def construct_exemplar_set(self, images, n_images ,m, transform):
        """Construct an exemplar set for image set

        Args:
            images: np.array containing images of a class
        """
        # Compute and cache features for each example
        loader = torch.utils.data.DataLoader(images, batch_size=self.batch_size/2,
                                       shuffle=True, num_workers=2)
        features = []
        start=time.time()
        for i, (indices, img, _) in enumerate(loader):
            x = Variable(img).cuda()
            feature = self.feature_extractor(x)
            feature = feature / torch.norm(feature, p=2)

            # features.append(feature.data.cpu().numpy()[0])
            if features == []:
                features = feature.data.cpu().numpy()
            else:
                features = np.vstack((features,feature.data.cpu().numpy()))

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean) # Normalize
        end=time.time()
        print("feature extraction: ",end-start);
        start=end;

        exemplar_set = [];
        exemplar_features = []; # list of Variables of shape (feature_size,)
        check_indexes=[];
        phi = features
        mu = class_mean
        S=np.zeros(phi.shape[1]);

        for k in range(m):
            mu_p_cuda=(torch.from_numpy(1.0/(k+1) * (phi + S))).cuda()
            mu_p_cuda = mu_p_cuda / torch.norm(mu_p_cuda, p=2)
            dist_mu_p_cuda = torch.from_numpy(mu).cuda() - mu_p_cuda.float()
            dist_mu_p_cuda = torch.sum((dist_mu_p_cuda) **2, 1)
            dist_mu_p_cuda = torch.sqrt(dist_mu_p_cuda)
            _ , sort_list = torch.sort(dist_mu_p_cuda)

            for i in sort_list:
                if(i not in check_indexes):
                    exemplar_set.append(n_images[i])
                    exemplar_features.append(features[i])
                    check_indexes.append(i)
                    break;
                # k+=1;
            S += exemplar_features[k];

        end=time.time()
        print("exemplar append: ",end-start);
        self.exemplar_sets.append(np.array(exemplar_set))
    
    def centroid_distance(self, images, transform):
        loader = torch.utils.data.DataLoader(images, batch_size=self.batch_size/2,
                                       shuffle=True, num_workers=2)
        features = []
        start=time.time()
        for i, (indices, img, _) in enumerate(loader):
             # x = Variable(transform(Image.fromarray(img)), volatile=True).cuda()
            x = Variable(img).cuda()
            feature = self.feature_extractor(x)
            feature = feature / torch.norm(feature, p=2)

            # features.append(feature.data.cpu().numpy()[0])
            if features == []:
                features = feature.data.cpu().numpy()
            else:
                features = np.vstack((features,feature.data.cpu().numpy()))

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean) # Normalize
        
        phi = features
        mu = class_mean

        mu_p_cuda=(torch.from_numpy((phi))).cuda()
        mu_p_cuda = mu_p_cuda / torch.norm(mu_p_cuda, p=2)
        dist_mu_p_cuda = torch.from_numpy(mu).cuda() - mu_p_cuda.float()
        dist_mu_p_cuda = torch.sum((dist_mu_p_cuda) **2, 1)
        dist_mu_p_cuda = torch.sum(torch.sqrt(dist_mu_p_cuda)) / len(images)

        phi = features = []

        return dist_mu_p_cuda


    def combine_dataset_with_exemplars(self, dataset):
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(exemplar_images, exemplar_labels)


    def update_representation(self, dataset, transform):

        self.compute_means = True

        # Increment number of weights in final fc layer
        classes = list(set(dataset.train_labels))
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.increment_classes(len(new_classes))
        self.cuda()
        print ("%d new classes" % (len(new_classes)))

        # # Form combined training set
        # self.combine_dataset_with_exemplars(dataset)

        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                               shuffle=True, num_workers=2)

        # Store network outputs with pre-update parameters
        q = torch.zeros(len(dataset), self.n_classes).cuda()
        for indices, images, labels in loader:
            images = Variable(images).cuda()
            indices = indices.cuda()
            g = F.sigmoid(self.forward(images))
            q[indices] = g.data
        q = Variable(q).cuda()

        # Run network training
        optimizer = self.optimizer

        for epoch in range(num_epochs):
            for i, (indices, images, labels) in enumerate(loader):
                images = Variable(images).cuda()
                labels = labels.long()              #addes for tensor type
                labels = Variable(labels).cuda()
                indices = indices.cuda()

                optimizer.zero_grad()
                g = self.forward(images)
                loss = self.cls_loss(g, labels)
            
                # centordi loss
                # if (i+1) % 30 == 0 or i==0:
                # if i+1:
                #     start = time.time()
                #     c_distance=[];
                #     for k in range(self.n_classes):
                #         images = dataset.get_image_class(k)
                #         images_label = np.ones((images.shape[0])) * k
                #         images = numpy_2_dataset(images, images_label, transform)        
                #         dist_val = self.centroid_distance(images,transform)
                #         c_distance.append(dist_val)
                    
                #     end=time.time()
                #     print("centroid_distance extraction: ",end-start);

                #     c_distance=np.asarray(c_distance)
                #     labels_index = labels.data.cpu().numpy()
                #     c_distance=np.sum(c_distance[labels_index])
                #     c_distance=c_distance/(self.n_classes*batch_size)
                #     # b = 10
                #     # c_distance=1/(b*self.n_classes)*1/batch_size*c_distance
                #     c_loss=Variable(torch.from_numpy(np.array([c_distance]))).cuda()
                #     c_loss=c_loss.float()
                #     loss.data = loss.data+c_loss.data
                #     print('C_loss: %.4f'%(c_loss.data[0]))

                loss.backward()
                optimizer.step()

                if (i+1) % 10 == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                           %(epoch+1, num_epochs, i+1, len(dataset)//self.batch_size, loss.data[0]))
