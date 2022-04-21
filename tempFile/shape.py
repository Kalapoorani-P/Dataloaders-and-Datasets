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
        self.maxWidth = 0
        self.maxHeight = 0
        self.batch=[]
        for labelno,label in enumerate(self.Imageclass):
            for images in os.listdir(self.path+"\\"+label):
                self.allImagePath.append(self.path + "\\" + label + "\\" + images)
                self.allLabels.append(labels_map[label])
        
    def findingMaxWidthAndHeight(self):
        currBatchwidth , currBatchHeight = [],[]
        for ind in self.batch:
            image = cv2.imread(self.allImagePath[ind])
            height,width= image.shape[:2]
            currBatchwidth.appe
            currBatchHeight.append(height)
        # print(currBatchHeight,currBatchwidth)
        # setting maxWidth for current Batch
        self.maxWidth = max(currBatchwidth)
        # setting max height for current Batch
        self.maxHeight = max(currBatchHeight)
        # print("function",dataset.maxWidth,dataset.maxHeight)
       
    def padding_image(self,img,imgWidth,imgHeight):
        # width difference
        widthDiff = self.maxWidth - imgWidth
        # height Difference 
        heightDiff = self.maxHeight - imgHeight 
        print((widthDiff, heightDiff),(self.maxWidth, self.maxHeight))
        padTop = heightDiff // 2
        padBottom = heightDiff - padTop 
        padLeft = widthDiff // 2
        padRight = widthDiff - padLeft 
        
        # creating border 
        paddedImage = cv2.copyMakeBorder(img, padTop,padBottom,padLeft,padRight,cv2.BORDER_CONSTANT,value=0)
       
        return paddedImage
    
    def normalize(self,imgNorm):

        return 2*((imgNorm - np.amin(imgNorm ))/ (np.amax(imgNorm) - np.amin(imgNorm))) - 1

    def tranforms(self,img):
        return cv2.resize(img,(500,500))
    def __getitem__(self,index) :
        print(index)
        img = cv2.imread(self.allImagePath[index])
        
        label = self.allLabels[index]

        height,width = img.shape[:2]
        # self.findingMaxWidthAndHeight()
        # img = self.padding_image(img,width,height) 
        img =self.tranforms(img)

        img = self.normalize(img)
        
        return img.shape,label
    def __len__(self) :
        return len(self.allImagePath)

def padding(dataset,batch):
    print(batch)
    currBatchwidth , currBatchHeight = [],[]
    for ind in batch:
        image = cv2.imread(dataset.allImagePath[ind])
        height,width= image.shape[:2]
        currBatchwidth.append(width)
        currBatchHeight.append(height)
    print(currBatchHeight,currBatchwidth)
    # setting maxWidth for current Batch
    dataset.maxWidth = max(currBatchwidth)
    # setting max height for current Batch
    dataset.maxHeight = max(currBatchHeight)
    print("function",dataset.maxWidth,dataset.maxHeight)

# Custome Batch Sampler 

class CustomeBatchSampler(Sampler):
    def __init__(self,dataset,batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size 
        self.evenIndexList = list(range(0,len(self.dataset),2))
        self.oddIndexList = list(range(1,len(self.dataset),2))
      
        
    def __iter__(self):
        # shuffling odd indices
        random.shuffle(self.oddIndexList)
        # shuffling even indices
        random.shuffle(self.evenIndexList)
    
        oddIndexBatch,evenIndexBatch = [], []

        # OddIndex Batching 
        batch1=[]
        for ind in self.oddIndexList:
            batch1.append(ind)
            if len(batch1) ==  self.batch_size:
                oddIndexBatch.append(batch1) 
                batch1 = []
        if len(batch1) > 0:
            oddIndexBatch.append(batch1)

        # EvenIndex batching 
        batch2 = []
        for ind in self.evenIndexList:
            batch2.append(ind)
            if len(batch2) == self.batch_size :
                evenIndexBatch.append(batch2)
                batch2 = []
        if len(batch2) > 0:
                evenIndexBatch.append(batch2)

        # Combined - Odd abd Even batches
        oddEvenBatch = list(oddIndexBatch+evenIndexBatch)

        random.shuffle(oddEvenBatch)

        for batch in oddEvenBatch:
            # print(self.dataset.batch)
            yield batch
        

    def __len__(self):
        return len(self.data)//self.batch_size
        
def  collate_fn(batch):
    print(batch)
    image_list,label_list = [],[]
    for img,label in batch:
        image_list.append(img)
        label_list.append(label)
    image_list = np.array(image_list)
    image_list = torch.as_tensor(image_list)
    label_list = torch.tensor(label_list)
    return image_list,label_list


if __name__ == '__main__':
    path = "C:\\Users\\test\\Desktop\\kp\\DatasetAndDataloader\\Data"
    myData = MyDataSet(path)
   
    myBatchSampler = CustomeBatchSampler(myData,batch_size=12)

    dataloader = DataLoader(myData,batch_sampler=myBatchSampler,collate_fn=collate_fn, num_workers=2)
    # print(dataloader)
    
    for (img,traget) in dataloader:
        print((img,traget))