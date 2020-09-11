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

#from sklearn.model_selection import train_test_split

#epochs= 2

#Transformacion de datos
#Convierte las imagenes a tensoress y los normaliza
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

#Descarga y guarda 
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader) #cada que se inicie el programa se accede a un batch diferente
images, labels = dataiter.next() #cada que se inicie el programa se accede a un batch diferente


print(images.shape)#torch.Size([64, 1, 28, 28])
print(labels.shape)#torch.Size([64])

#plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
#plt.show()

#diseñando la red
class RED(nn.Sequential):
    def __init__(self):
        super(RED, self).__init__()
        self.linear1 = nn.Linear(784,64)
        #self.linear2 = nn.Linear(250,100)
        self.linear3 = nn.Linear(64,10)
    
    def forward(self,X):
        X = torch.sigmoid(self.linear1(X))
        #X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)
 
modelo= RED() #Objeto 
print(modelo)
#definimos el optimizador

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1) #Aplana las imagenes a 64 * 784

logps = modelo(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', modelo[0].weight.grad)
loss.backward()
print('After backward pass: \n', modelo[0].weight.grad)


optimizer = optim.SGD(modelo.parameters(), lr=0.1)
time1 = time()
epochs = 20
for e in range(epochs):
    running_loss = 0 								#Establece en 0 la perdida total de esta epoca
    time0 = time()
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1) 	#view sirve para cambiar el tamaño del tensor (images.shape[0] = 64 x -1)
        											#Al inicio el tensor es de [64, 1, 28, 28] [imagenes,escala de grises,ancho,alto] (64 x 1 x 28 x 28 = 50176)
        											#Se cambiara el tamaño del tensor a uno de 64 x -1 (el -1 le dice a la computadora que ella acomplete el dato)
        											#Para hacer el cambio de tamaño tiene que caber todos los datos que tenias en la nueva forma 
        											
        											#Ejem. si tienes un tensor de una dimension con 16 datos, al reformarlo tiene que haber espacio suficiente para todos los datos
        											#Al cambiarlo a 2 dimensiones tendria que ser por lo menos de (2 x 8 = 16) o (4 x 4 = 16) pero si lo haces menor como (3 x 3) te marcara error
        											
        											#Entonces en este caso el -1 hace que la compu calcule la dimension faltante (64 x ? = 50176 ) -> (? = 50176/64) -> (? -> 784)
        											#Que este seria nuestro aplanado, algo diferente a keras que es mas intuitivo 
        #print(images.shape)
    
        # Training pass
        optimizer.zero_grad()						#Pone en 0 todos los gradiantes, al parecer se necesita hacer antes de la retropropagacion 
        
        output = modelo(images)						#Se introduce la informacion de un batch a la red neuronal y nos regresa el 
        print("output: ",output)
        print("labels: ",labels)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()					#va sumando los errores de cada lote (0-1). Lo esperado es que esto sea 0
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        print("time: ",round(time()-time0)/60,"min")
print("\nTraining Time (in minutes) =",(time()-time1)/60)


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


images, labels = next(iter(valloader))

img = images[0].view(1, 784)
with torch.no_grad():
    logps = modelo(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)




