from torch.utils.data import Dataset,DataLoader,Sampler
import torch
class Mydata:
    def __init__(self):

        self.numbers = list(range(0,15))
    def __getitem__(self,ind):

        print("GET ITEM :",ind,"      Worker_id - ",torch.utils.data.get_worker_info().id)
        return self.numbers[ind]
    def __len__(self):
        return len(self.numbers)
def collate_fn(batch):
    print("COLLATE FUNCTION",batch,)
    return torch.tensor(batch)
if __name__ == "__main__":
    Data = Mydata()
    dataloader = DataLoader(Data,batch_size=2,num_workers=2,collate_fn=collate_fn,prefetch_factor=2)
    for i,val in enumerate(dataloader):
        print(i,val)

    print("OVER")