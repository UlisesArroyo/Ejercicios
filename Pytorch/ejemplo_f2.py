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

class MyModel(nn.Sequential):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden1 = nn.Linear(784, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 10)


    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.hidden3(x)
        x = torch.log_softmax(x, dim=1)
        return x

model = MyModel()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


train_dataset = datasets.MNIST(root='PATH_TO_STORE_TRAINSET',train=True,transform=transforms.ToTensor())

val_dataset = datasets.MNIST(root='PATH_TO_STORE_TESTSET',train=False,transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=128,shuffle=False)
time1 = time() 

for epoch in range(1):
    train_loss = 0.
    val_loss = 0.
    train_acc = 0.
    val_acc = 0.
    time0 = time()  
    for data, target in train_loader:
        # Transform target to one-hot encoding, since Keras uses MSELoss
        #target = torch.zeros(data.size(0), 10).scatter_(1, target[:, None], 1.)
        
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        #print("output", output.shape)
        #print("target", target.shape)  
        train_loss += loss.item()



        train_acc += (torch.argmax(output, 1) == target).float().sum()
    with torch.no_grad():
        for data, target in val_loader:
            #target = torch.zeros(data.size(0), 10).scatter_(1, target[:, None], 1.)
        
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)

            val_loss += loss.item()
            val_acc += (torch.argmax(output, 1) == target).float().sum()
    
    train_loss /= len(train_loader)
    train_acc /= len(train_dataset)
    val_loss /= len(val_loader)
    val_acc /= len(val_dataset)

    print('Epoch {}, train_loss {}, val_loss {}, train_acc {}, val_acc {}'.format(epoch, train_loss, val_loss, train_acc, val_acc))
    print("tiempo: ",time()-time0,"(seg)")
time2 = time()        
print("\nTraining Time (in minutes) =",(time2-time1)/60)    

print("train_dataset: " ,train_dataset)
print("train_loader: " ,train_loader)
