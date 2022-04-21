
from torch.utils.data import Dataset,DataLoader,Sampler
import random

# Sequential Sampler 
class sequentialSampler(Sampler):

    def __init__(self,dataset):
        # print(data)
        self.dataset = dataset
    def __iter__(self):
        for index in range(len(self.dataset)):
            yield index 
    def __len__(self):
        return len(self.dataset)
# RandomSampler 
class randomSampler(Sampler):

    def __init__(self,dataset):
        self.dataset = dataset

    def __iter__(self):
        indices = range(len(self.dataset))
        random.shuffle(indices)
        for index in indices:
            yield index 

    def __len__(self):
        return len(self.data)

# OddSampler 
class oddSampler(Sampler):

    def __init__(self,dataset):
        self.dataset = dataset
        self.oddIndexList = list(range(1,len(self.dataset),2))

    def __iter__(self):
        random.shuffle(self.oddIndexList)
        for index in  self.oddIndexList:
            yield index

    def __len__(self):
        return len(self.oddIndexList)

# Even sampler
class evensampler(Sampler):

    def __init__(self,dataset):
        self. dataset = dataset
        self.evenIndexList = list(range(0,len(self.dataset),2))
    def __iter__(self):
        random.shuffle(self.evenIndexList)
        for index in  self.evenIndexList:
            yield index 
    def __len__(self):
        return len(self.evenIndexList)
