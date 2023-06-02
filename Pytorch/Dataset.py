import torchvision
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

class Dataset:
    def __init__(self,path:str,batch_size:int,shuffle:bool) -> None:
        self.path=path
        self.transform=transforms.Compose([
            transforms.Resize(size=(80,80)),
            transforms.ToTensor()
        ])
        self.dataset=ImageFolder(root=self.path,
                                 transform=self.transform,
                                 )
        self.classes=self.dataset.classes
        self.classes_idx=self.dataset.class_to_idx
        self.dataloader=DataLoader(dataset=self.dataset,
                                   batch_size=batch_size,shuffle=shuffle)
    
    def getSampleImage(self):
        torch.manual_seed(42)
        random_idx=random.randint(0,len(self.dataset))
        img,label=self.dataset[random_idx]
        permute=img.permute(1,2,0)
        plt.imshow(permute)
        plt.title(str(self.classes[label]))
        plt.axis("off")
        return (img,label)