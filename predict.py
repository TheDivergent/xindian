import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import glob
import scipy.io as sio
import os
import pandas as pd
#from seedata import MyDataset,TextCNN

#基本模型

class MyDataset(Dataset):
    def __init__(self,mat,label,mat_dim=3000):
        super(MyDataset,self).__init__()
        self.mat=mat
        self.label=label
        self.mat_dim=mat_dim

    def __len__(self):
        return len(self.mat)

    def __getitem__(self, index):
        idx=np.random.randint(0,5000-self.mat_dim)
        return torch.tensor(self.mat[index][:,:,idx:idx+self.mat_dim]),self.label[index]



class TextCNN(nn.Module):
    def __init__(self,kernel_num=30,kernel_size=[3,4,5],dropout=0.5):
        super(TextCNN, self).__init__()
        self.kernel_num=kernel_num
        self.kernel_size=kernel_size
        self.dropout=dropout

        self.convs=nn.ModuleList([nn.Conv2d(1,self.kernel_num,(kernel_size_,3000)) for kernel_size_ in kernel_size])
        self.dropout=nn.Dropout(self.dropout)
        self.linear=nn.Linear(3*self.kernel_num,1)

    def forward(self,x):
        convs=[nn.ReLU()(conv(x)).squeeze(3) for conv in self.convs]
        pool_out=[nn.MaxPool1d(block.shape[2])(block).squeeze(2) for block in convs]
        pool_out=torch.cat(pool_out,1)
        logits=self.linear(pool_out)

        return logits

#定义超参数、等
model=TextCNN()
BATCH_SIZE=30
EPOCHS=200
Learning_rate=0.0005
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


#读取训练集
test_mat=glob.glob('data/val/*.mat')
test_mat.sort()
test_mat=[sio.loadmat(x)['ecgdata'].reshape(1,12,5000) for x in test_mat]




test_pred=np.zeros(len(test_mat))
tta_count=20

print('Datasets:',len(test_mat))

#训练模型：
for fold_idx in range(10):
    Test_Loader=DataLoader(MyDataset(test_mat,torch.tensor([0]*len(test_mat))),batch_size=BATCH_SIZE,shuffle=False)
    model.load_state_dict(torch.load(f'model_{fold_idx}.pth'))
    for tta in range(tta_count):
        test_pred_list=[]
        for i,(x,y) in enumerate(Test_Loader):
            if device=='gpu':
                x=x.cuda()
                y=y.cuda()
            pred=model(x)
            print('pred:',pred)
            test_pred_list.append(nn.functional.sigmoid(pred).detach().numpy())
        print('test_pred_list: ',test_pred_list)
        test_pred+=np.vstack(test_pred_list)[:,0]

test_pred/=tta_count*10

test_path=glob.glob('data/val/*.mat')
test_path=[os.path.basename(x)[:-4] for x in test_path]
test_path.sort()

test_answer=pd.DataFrame({'name':test_path,'tag':(test_pred>0.5).astype(int)}).to_csv('answer.csv',index=False)
