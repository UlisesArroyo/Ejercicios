import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from sklearn.datasets import load_boston
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
class MyModel(nn.Sequential):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1     = nn.Linear(13, 32)
        self.bn1        = nn.BatchNorm1d(32)
        self.layer2     = nn.Linear(32, 16)
        self.bn2        = nn.BatchNorm1d(16)
        self.drop_layer = nn.Dropout(0.2)
        self.layer3     = nn.Linear(16, 1)


    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.bn2(self.layer2(x))
        x = torch.relu(self.drop_layer(x))
        #x = torch.relu(self.drop_layer(self.bn2(self.layer2(x))))
        x = self.layer3(x)
        #x = nn.Linear(self.layer3(x))
        return x

model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

"""
train_dataset = datasets.MNIST(root='PATH_TO_STORE_TRAINSET',train=True,transform=transforms.ToTensor())
val_dataset = datasets.MNIST(root='PATH_TO_STORE_TESTSET',train=False,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=128,shuffle=False)
"""
boston = load_boston()
X,y   = (boston.data, boston.target)
dim = X.shape[1]
print("dim: ",dim)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
num_train = X_train.shape[0]



train_dataset 	= TensorDataset(torch.from_numpy(X_train).clone().float(), torch.from_numpy(y_train).clone().float())
val_dataset		= TensorDataset(torch.from_numpy(X_test).clone().float(), torch.from_numpy(y_test).clone().float())

train_loader 	= DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
val_loader		= DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)

time1 = time() 

for epoch in range(32):
    train_loss = 0.
    val_loss = 0.
    time0 = time()  
    for data, target in train_loader:
        
        optimizer.zero_grad()
        #print("data: ",data.shape)
        output = model(data)
        loss = criterion(output.view(output.size(0)), target)
        loss.backward()
        optimizer.step()
        #print("output", output[1].shape)
        #print("target", target.shape)  
        train_loss += loss.item()

    with torch.no_grad():
        for data, target in val_loader:
            #target = torch.zeros(data.size(0), 10).scatter_(1, target[:, None], 1.)
        
            output = model(data)
            loss = criterion(output.view(output.size(0)), target)

            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print('Epoch {}, train_loss {}\t, val_loss {},\t'.format(epoch, train_loss, val_loss),end="")
    print("        ||tiempo: ",time()-time0,"(seg)")
time2 = time()        
print("\nTraining Time (in seconds) =",(time2-time1))   



#model.predict(np.reshape(X_valid[42], [1, 13]))

py = model(torch.DoubleTensor(X_train))
plt.plot(y_train, py.detach().numpy(), '+')
plt.xlabel('Actual value of training set')
plt.ylabel('Prediction') 