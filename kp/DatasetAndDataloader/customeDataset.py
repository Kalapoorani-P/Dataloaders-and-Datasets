import torch 
from torch.utils.data import Dataset,DataLoader,Sampler
from torchvision import datasets,transforms
import numpy as np
import cv2 
import os
import random
labels_map={"aeroplane":0, "cat":1,"dog":2,"person":3,"train":4,}

class MyDataSet(Dataset) :
    def __init__(self,path):
        self.path = path
        self.allImagePath = []
        self.allLabels = []
        self.Imageclass = os.listdir(self.path)
        # print(self.labelOfImage)
        # print(self.labeImage)
        for labelno,label in enumerate(self.Imageclass):
            for images in os.listdir(self.path+"\\"+label):
                self.allImagePath.append(self.path + "\\" + label + "\\" + images)
                self.allLabels.append(labels_map[label])
        self.transforms = lambda img: cv2.resize(img,(500,500))
        self.normalize = lambda imgNorm: 2*((imgNorm - np.amin(imgNorm ))/ (np.amax(imgNorm) - np.amin(imgNorm))) - 1
    def __getitem__(self,index) :
        # print("GetItem function",index)
        img = cv2.imread(self.allImagePath[index])
        label = self.allLabels[index]
        img = self.transforms(img)
        img = self.normalize(img)
        return img,label
    def __len__(self) :
        return len(self.allImagePath)
class CustomeSampler(Sampler):
    def __init__(self,dataset,SequenceType="s"):
         self.dataset = dataset
         self.SequenceType = SequenceType.lower()
    def __iter__(self):
        # print("Starting Sampler")
        if self.SequenceType =='s':
            for i in range(len(self.dataset)):
                yield i
        elif self.SequenceType =="r":
            shuffledIndexList = random.sample(range(len(self.dataset)),len(self.dataset))
            # print(shuffledIndexList)
            for i in shuffledIndexList :
                # print("Sampler Function",i)
                yield i
        else:
            print("Invalid SequenceType")
    def __len__(self):
        return len(self.dataset) 

class CustomeBatchSampler:
    def __init__(self,sampler,batch_size=1):
        
        self.sampler = sampler
        self.batch_size = batch_size 
        
    def __iter__(self):
        print("CustomeBatchSampler")
        batch = []
        # sampler=iter(self.sampler)
        # while 1:
        #     try:
        #         batch.append(next(sampler))
        #        
        #         if len(batch)==self.batch_size:
        #             yield batch 
        #             batch = []
                
        #     except:
        #         if len(batch) > 0 and not self.drop_last:
        #             yield batch
        #         break
        for indx in self.sampler:
            # print("calling sampler",count,indx)
        
            batch.append(indx)
            # print("Batch",batch)
            if len(batch) == self.batch_size:
                    print("Batch",batch)
                    yield batch 
                    batch = []

        if len(batch) > 0:
                    yield batch


def  collate_fn(batch):
    # print("collate",batch)
    image_list,label_list = [],[]
    for img,label in batch:
        image_list.append(img)
        label_list.append(label)
    image_list=np.array(image_list)
    image_list = torch.as_tensor(image_list)
    label_list = torch.tensor(label_list)
    return image_list,label_list


if __name__ == '__main__':
    path = "C:\\Users\\test\\Desktop\\kp\\DatasetAndDataloader\\Data"
    myData = MyDataSet(path)
    print(myData[0][0].shape)
    print(len(myData))

    mySampler = CustomeSampler(myData,"r")

    myBatchSampler = CustomeBatchSampler(mySampler,batch_size=12)

    dataloader = DataLoader(myData,batch_sampler=myBatchSampler, collate_fn=collate_fn, num_workers=1)
    print(dataloader)
    # print(next(iter(dataloader)))
    for (img,traget) in dataloader:
        print((img.shape,traget))