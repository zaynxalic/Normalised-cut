# extract the pictures from directories
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

class preprocessing:
    def __init__(self):
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

    def initialise(self):
        """
        Initialise the CIFAR10 datasets.
        """
        transform =  transforms.Compose([transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                shuffle=False, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                shuffle=False, num_workers=2)
        return trainloader, testloader

    def detransform(self,img):
        # img = img / 2 + 0.5     # unnormalize
        img = img.numpy()
        return img

    def tensorimage(self):
        """
        show the images with labels of horse, deer, and airplane
        """
        trainloader, _ = self.initialise()  
        dataiter = iter(trainloader)
        dict = {}
        pic_idx = [3,3,1]
        while len(dict.keys()) != 3: #740
            images, labels = dataiter.next()
            if  labels.item() == 7:
                pic_idx[0] -= 1
                if pic_idx[0] == 0:
                    dict.setdefault(labels.item(), images)

            if labels.item() == 4:
                pic_idx[1] -= 1
                if pic_idx[1] == 0:
                    dict.setdefault(labels.item(), images)

            if labels.item() == 0:
                pic_idx[2] -= 1
                if pic_idx[2] == 0:
                    dict.setdefault(labels.item(), images)

        img = torch.empty(3,3,32,32)
        for idx,v in enumerate(dict.values()):
            img[idx] = v 
        
        img = self.detransform(img) # denormalise

        return img 
