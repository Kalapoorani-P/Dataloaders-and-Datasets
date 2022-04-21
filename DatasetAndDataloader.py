import torch 
from torch.utils.data import Dataset,DataLoader,get_worker_info
import numpy as np
import cv2 
import os
import random
from sampler import *

labels_map={"aeroplane":0, "cat":1,"dog":2,"person":3,"train":4}


class MyDataSet(Dataset) :
    def __init__(self,path):
        self.path = path
        self.allImagePath = []
        self.allLabels = []
        self.Imageclass = os.listdir(self.path)
        for labelno,label in enumerate(self.Imageclass):
            for images in os.listdir(self.path+"\\"+label):
                self.allImagePath.append(self.path + "\\" + label + "\\" + images)
                self.allLabels.append(labels_map[label])
        self.sortByHeightWidth()

    
    def sortByHeightWidth(self):

        self.allImagePath = sorted(self.allImagePath,key=lambda x: cv2.imread(x).shape)


    def normalize(self,imgNorm):

        return 2*((imgNorm - np.amin(imgNorm ))/ (np.amax(imgNorm) - np.amin(imgNorm))) - 1

    
    def __getitem__(self,index) :
        print("INSIDE GET ITEM",torch.utils.data.get_worker_info())
        # print(index)
        img = cv2.imread(self.allImagePath[index])
        
        label = self.allLabels[index]

        img = self.normalize(img)

        return img,label

    def __len__(self) :

        return len(self.allImagePath)



class CustomeBatchSampler:
    def __init__(self,dataset,sequenceType,batch_size=1,drop_last= False):

        self.dataset = dataset
        self.batch_size = batch_size 
        self.drop_last = drop_last
        self.sequenceType = sequenceType
        if self.sequenceType == "s" or self.sequenceType == None:
            self.sampler = sequentialSampler(self.dataset)
        elif self.sequenceType == "r":
            self. sampler = randomSampler(self.dataset)
        elif self.sequenceType == "oe":
            self.sampler1 = oddSampler(self.dataset)
            self.sampler2 = evensampler(self.dataset)
        
    def __iter__(self):
        print("INSIDE ITER OF SAMPLER")

        if self.sequenceType == "oe":
            oddIndexBatch,evenIndexBatch = [], []
            batch1=[]
            for ind in self.sampler1:
                batch1.append(ind)
                if len(batch1) ==  self.batch_size:
                    oddIndexBatch.append(batch1) 
                    batch1 = []
            if len(batch1) > 0 and not self.drop_last :
                oddIndexBatch.append(batch1)

            # EvenIndex batching 
            batch2 = []
            for ind in self.sampler2:
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
        else:
            batch=[]
            for index in self.sampler:
                batch.append(index)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = [] 
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            return  ((len(self.sampler))//self.batch_size)
        else:
            if self.sequenceType=="oe":
                return (len(self.sampler1)+self.batch_size-1)//2 + (len(self.sampler1)+self.batch_size-1)//2
            else:
                return len(self.sampler)//self.batch_size
           

def padding_image(img,maxWidth,maxHeight,imgWidth,imgHeight):

    # width Difference
    widthDiff = maxWidth - img.shape[1]
    # height Difference 
    heightDiff = maxHeight - img.shape[0]

    padTop = heightDiff // 2
    padBottom = heightDiff - padTop 
    padLeft = widthDiff // 2
    padRight = widthDiff - padLeft 
    
    # creating border 
    paddedImage = cv2.copyMakeBorder(img, padTop,padBottom,padLeft,padRight,cv2.BORDER_CONSTANT,value=0)
    
    return paddedImage
def random_padding(img,maxWidth,maxHeight,image):
   
    newImg = np.full((maxHeight,maxWidth,image.shape[-1]),(0,0,0),dtype=np.float64)
    # width Difference
    widthDiff = maxWidth - img.shape[1]
    # height Difference 
    heightDiff = maxHeight - img.shape[0]

    randwidth= random.randint(0,widthDiff)
    randHeight= random.randint(0,heightDiff)

    newImg[randHeight:randHeight+image.shape[0],randwidth:randwidth+image.shape[1]] = image

    return newImg

def FindingMaxHeightWidth(ImgList):
    maxWidth=maxHeight = 0
    for image in ImgList:
        height,width= image.shape[:2]
        maxHeight = height if height > maxHeight else maxHeight 
        maxWidth = width if width > maxWidth else maxWidth
    # newImgList = [ padding_image(img,maxWidth,maxHeight,img.shape[1],img.shape[0]) for img in ImgList ]
    newImgList = [ random_padding(img,maxWidth,maxHeight,img) for img in ImgList]
    return newImgList


    
def  collate_fn(batch):
    print("INSIDE COLLATE FUNCTION",get_worker_info())
  
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
   
    myBatchSampler = CustomeBatchSampler(myData,sequenceType="oe",batch_size=12)

    dataloader = DataLoader(myData,batch_sampler=myBatchSampler,collate_fn=collate_fn, num_workers=2)

    print(len(dataloader))

    for index,(img,traget) in enumerate(dataloader):
        print(f" Batch - {index}",(img.shape,traget))

print("DONE")