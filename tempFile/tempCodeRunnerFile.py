import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import os

transform = transforms.Compose([transforms.Resize([500,500])])
path = "C:\\Users\\test\\Desktop\\kp\\DatasetAndDataloader\\Data"
myData = datasets.ImageFolder(path,transforms=transform)
