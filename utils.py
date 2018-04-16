import torch
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from data_loader import *
class utils(object):
	def __init__(self,datasets_name=None):
		self.RGB=True
		self.datasets_name=datasets_name
		self.train_set=[]
		self.train_transform=None
		self.test_set=[]
		self.test_transform=None
		self.label_name=[]
		self.dataset_manage();

	def dataset_manage(self):

		transform_test = transforms.Compose([
			transforms.ToTensor()
		])

		print("Loading training examples of ",self.datasets_name);
		if (self.datasets_name == "mnist"):
			self.train_transform = transform_test
			self.test_transform = transform_test
			self.train_set = ori_MNIST(root='./data',
								train=True,
								download=True,
								transform=self.train_transform)
			self.test_set = ori_MNIST(root='./data',
								train=False,
								download=True,
								transform=self.test_transform)
			self.RGB=False;

		elif (self.datasets_name == "noise_mnist"):
			self.train_transform = transform_test
			self.train_set = noi_MNIST(root='./data',
								train=True,
								download=True,
								transform=self.train_transform)
			self.RGB=False;

		elif (self.datasets_name == "mix_mnist"):
			self.train_transform = transform_test
			self.test_transform = transform_test
			ori_train_set = ori_MNIST(root='./data',
								train=True,
								download=True,
								transform=self.train_transform)
			ori_test_set = ori_MNIST(root='./data',
								train=False,
								download=True,
								transform=self.test_transform)

			noi_train_set = noi_MNIST(root='./data',
								train=True,
								download=True,
								transform=self.train_transform)

			self.train_set = combined_MNIST(root='./data',
								train=True,
								download=True,
								transform=self.train_transform,
								mnistdata=ori_train_set,
								nmnistdata=noi_train_set,
								mix=0.5)
			self.test_set = combined_MNIST(root='./data',
								train=False,
								download=True,
								transform=self.test_transform,
								mnistdata=ori_test_set,
								nmnistdata=noi_train_set,
								mix=0)
			self.RGB=False;

		elif (self.datasets_name == "cifar10"):
			self.train_transform = transforms.Compose([
				# transforms.Resize(224),
				transforms.ToTensor(),
			])

			self.test_transform = transforms.Compose([
				# transforms.Resize(224),
				transforms.ToTensor(),
			])
			self.train_set = ori_CIFAR10(root='./data',
								train=True,
								download=True,
								transform=self.train_transform)
			self.test_set = ori_CIFAR10(root='./data',
								train=False,
								download=True,
								transform=self.test_transform)

		elif (self.datasets_name == "cifar100"):
			self.train_transform = transforms.Compose([
				transforms.ToTensor(),
			])

			self.test_transform = transforms.Compose([
				transforms.ToTensor(),
			])
			self.train_set = ori_CIFAR100(root='./data',
								train=True,
								download=True,
								transform=self.train_transform)
			self.test_set = ori_CIFAR100(root='./data',
								train=False,
								download=True,
								transform=self.test_transform)
		
		elif (self.datasets_name == "cifar10a"):
			self.train_transform = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

			self.test_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			self.train_set = ori_CIFAR10(root='./data',
								train=True,
								download=True,
								transform=self.train_transform)
			self.test_set = ori_CIFAR10(root='./data',
								train=False,
								download=True,
								transform=self.test_transform)

		elif (self.datasets_name == "cifar10g"):
			self.train_transform = transforms.Compose([
				transforms.Resize(32),
				# transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

			self.test_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			cifar10_train_set = ori_CIFAR10(root='./data',
								train=True,
								download=True,
								transform=self.train_transform)
			cifar10_test_set = ori_CIFAR10(root='./data',
								train=False,
								download=True,
								transform=self.test_transform)
			google_image = cifar_ImageFolder(root='./data/web',
								transform=self.train_transform)

			self.train_set = combine_cifar(root='./data',
								train=True,
								ori_cifar=cifar10_train_set,
								web_cifar=google_image,
								download=False,
								transform=self.train_transform)
			self.test_set = combine_cifar(root='./data',
								train=False,
								ori_cifar=cifar10_test_set,
								web_cifar=google_image,
								download=False,
								transform=self.test_transform)
			print('for test stupid')
		else:
			print("Wrong d_sets Bitch")