"""
baseline：十折交叉验证，用TEXTCNN模型训练
"""
import codecs,glob,os
import numpy  as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

import matplotlib.pyplot as plt

import scipy.io as sio


from sklearn.model_selection import StratifiedKFold

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#读取数据
train_mat=glob.glob('data/train/*.mat')
train_mat.sort()
train_mat=[sio.loadmat(x)['ecgdata'].reshape(1,12,5000) for x in train_mat]

test_mat=glob.glob('data/test/*.mat')
test_mat.sort()
test_mat=[sio.loadmat(x)['ecgdata'].reshape(1,12,5000) for x in test_mat]

train_df=pd.read_csv('data/trainreference.csv')
train_df['tag']=train_df['tag'].astype(np.float32)


plt.plot(range(5000),train_mat[0][0][0])
plt.plot(range(5000),train_mat[0][0][1])
plt.plot(range(5000),train_mat[0][0][3])
#plt.show()

#建立数据集类
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


#定义模型TextCNN
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


model=TextCNN()
BATCH_SIZE=30
EPOCHS=200
Learning_rate=0.0005
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

#nn.summary(model,(64,1,9,3000))
print(model)


#训练模型
skf=StratifiedKFold(n_splits=10)

fold_idx=0
for trn_idx,val_idx in skf.split(train_mat,train_df['tag'].values):
    Train_Loader=DataLoader(MyDataset(np.array(train_mat)[trn_idx],torch.tensor(train_df['tag'].values[trn_idx])),batch_size=BATCH_SIZE,shuffle=True)
    Val_Loader=DataLoader(MyDataset(np.array(train_mat)[val_idx],torch.tensor(train_df['tag'].values[val_idx])),batch_size=BATCH_SIZE,shuffle=True)
    model=TextCNN()

    optimizer=optim.Adam(params=model.parameters(),lr=Learning_rate)
    criterion=nn.BCEWithLogitsLoss()

    Test_best_Acc=0
    for epoch in range(0,EPOCHS):
        Train_Loss,Test_Loss=[],[]
        Train_Acc,Test_Acc=[],[]
        model.train()
        for i,(x,y) in enumerate(Train_Loader):
            if device=='gpu':
                x=x.cuda()
                y=y.cuda()

            pred=model(x)
            loss=criterion(pred.view(y.shape),y)
            Train_Loss.append(loss.item())

            pred=(nn.functional.sigmoid(pred)>0.5).int()
            Train_Acc.append((pred.numpy()==y.numpy()).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        for i,(x,y) in enumerate(Val_Loader):
            if device=='gpu':
                x=x.cuda()
                y=y.cuda()

            pred=model(x)
            Test_Loss.append(criterion(pred.view(y.shape),y).item())
            pred=(nn.functional.sigmoid(pred)>0.5).int()
            Test_Acc.append((pred.numpy()==y.numpy()).mean())

        if epoch%10==0:
            print('Epoch: [{}/{}] TrainLoss/TestLoss: {:.4f}/{:.4f} TrainAcc/TestAcc: {:.4f}/{:.4f}'.format(epoch+1,EPOCHS,
                np.mean(Train_Loss),np.mean(Test_Loss),np.mean(Train_Acc),np.mean(Test_Acc)))

        if Test_best_Acc<np.mean(Test_Acc):
            print(f'Fold {fold_idx} Acc improve from {Test_best_Acc} to {np.mean(Test_Acc)} Save Model...')
            torch.save(model.state_dict(),f'model_{fold_idx}.pth')
            Test_best_Acc=np.mean(Test_Acc)

    fold_idx+=1

