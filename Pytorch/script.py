import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from torch import nn
import torch
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

class Model(nn.Module):
    def __init__(self,input_shape:int,hidden_unit:int,output_shape:int) -> None:
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_unit,
                    kernel_size=3,
                    padding=1,
                    stride=1,),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=hidden_unit,
                    out_channels=hidden_unit,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25600,out_features=hidden_unit),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_unit,out_features=output_shape),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x=self.conv(x)
        x=self.classifier(x)
        return x