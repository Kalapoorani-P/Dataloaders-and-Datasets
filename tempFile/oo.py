import torch 
from torch.utils.data import Dataset,DataLoader,Sampler
import numpy as np
import cv2 
import os
import random

labels_map={"aeroplane":0, "cat":1,"dog":2,"person":3,"train":4}


class MyDataSet(Dataset) :
    def __init__(self,path):
        self.path = path
        self.allImagePath = []
        self.allLabels = []
        self.Imageclass = os.listdir(self.path)
        self.maxWidth = 0
        self.maxHeight = 0
        for labelno,label in enumerate(self.Imageclass):
            for images in os.listdir(self.path+"\\"+label):
                self.allImagePath.append(self.path + "\\" + label + "\\" + images)
                self.allLabels.append(labels_map[label])
       
    def normalize(self,imgNorm):

        return 2*((imgNorm - np.amin(imgNorm ))/ (np.amax(imgNorm) - np.amin(imgNorm))) - 1

    
    def __getitem__(self,index) :
        # print(index)
        img = cv2.imread(self.allImagePath[index])
        
        label = self.allLabels[index]

        img = self.normalize(img)

        return img,label

    def __len__(self) :

        return len(self.allImagePath)

# Custome Batch Sampler 
class CustomeBatchSampler(Sampler):
    def __init__(self,dataset,sequenceType,batch_size=1,drop_last= False):

        self.dataset = dataset
        self.batch_size = batch_size 
        self.drop_last = drop_last
        self.sequenceType = sequenceType
        self.evenIndexList = list(range(0,len(self.dataset),2))
        self.oddIndexList = list(range(1,len(self.dataset),2))
      
    def __iter__(self):
        # sequential 
        if self.sequenceType == "s":
            self.NormalBatching(list(range(len(self.dataset))))

        # random 
        elif self.sequenceType == "r":
            self.NormalBatching(random.shuffle(list(range(len(self.dataset)))))

        # Odd even
        elif self.sequenceType == "oe":
            self.OddEvenBatching()
    def NormalBatching(self,sample):
        batch=[]
        for ind in sample:
            batch.append(ind)
            if len(batch)==self.batch_size :
                print(batch)
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch  
    def OddEvenBatching(self):
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
        if len(batch1) > 0 and not self.drop_last :
            oddIndexBatch.append(batch1)

        # EvenIndex batching 
        batch2 = []
        for ind in self.evenIndexList:
            batch2.append(ind)
            if len(batch2) == self.batch_size :
                evenIndexBatch.append(batch2)
                batch2 = []
        if len(batch2) > 0 and not self.drop_last:
                evenIndexBatch.append(batch2)
    
        # Combined - Odd and Even batches
        oddEvenBatch =oddIndexBatch+evenIndexBatch

        random.shuffle(oddEvenBatch)

        for batch in oddEvenBatch :
            yield batch

    def __len__(self):
        if self.drop_last:
            return  ((len(self.dataset))//self.batch_size)
        else:
            if self.sequenceType == "oe":
                return (len(self.oddIndexList)+self.batch_size -1)//self.batch_size + (len(self.evenIndexList)+self.batch_size-1)//self.batch_size

            elif self.sequenceType in ["r","s"] :
                return (len(self.oddIndexList) + self.batch_size-1) // self.batch_size


def padding_image(img,maxWidth,maxHeight,imgWidth,imgHeight):

    # width Difference
    widthDiff = maxWidth - imgWidth
    # height Difference 
    heightDiff = maxHeight - imgHeight 

    padTop = heightDiff // 2
    padBottom = heightDiff - padTop 
    padLeft = widthDiff // 2
    padRight = widthDiff - padLeft 
    
    # creating border 
    paddedImage = cv2.copyMakeBorder(img, padTop,padBottom,padLeft,padRight,cv2.BORDER_CONSTANT,value=0)
    
    return paddedImage

def FindingMaxHeightWidth(ImgList):
    maxWidth=maxHeight = 0
    for image in ImgList:
        height,width= image.shape[:2]
        maxHeight = height if height > maxHeight else maxHeight 
        maxWidth = width if width > maxWidth else maxWidth
    newImgList = [ padding_image(img,maxWidth,maxHeight,img.shape[1],img.shape[0]) for img in ImgList ]
    return newImgList

    
def  collate_fn(batch):
  
    transposed = zip(*batch)

    for sample in transposed:
        if  isinstance(sample[0],np.ndarray):
            image_list=FindingMaxHeightWidth(sample)
            image_list = torch.as_tensor(np.array(image_list))
        
        if isinstance(sample[0],int):
            label_list = torch.tensor(sample)
            
    return image_list,label_list


if __name__ == '__main__':

    path = "C:\\Users\\test\\Desktop\\kp\\DatasetAndDataloader\\Data"
    myData = MyDataSet(path)
    myBatchSampler = CustomeBatchSampler(myData,sequenceType="s",batch_size=12, drop_last = True)

    dataloader = DataLoader(myData,batch_sampler=myBatchSampler,collate_fn=collate_fn, num_workers=3)

    print(len(dataloader))

    for index,(img,traget) in enumerate(dataloader):
        print(f" Batch - {index}",(img.shape,traget))