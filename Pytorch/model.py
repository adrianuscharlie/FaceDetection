from torch import nn
from torch.utils.data import DataLoader
import torch
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