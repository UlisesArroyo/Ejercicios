import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from time import time


(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

#X_train.shape

#y_train.shape

#y_train[0:12]

plt.figure(figsize=(5,5))
for k in range(12):
    plt.subplot(3, 4, k+1)
    plt.imshow(X_train[k], cmap='Greys')
    plt.axis('off')
plt.tight_layout()
plt.show()

#X_valid.shape

#y_valid.shape

plt.imshow(X_valid[0], cmap='Greys')

#X_valid[0]

#y_valid[0]

X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')

X_train /= 255
X_valid /= 255

#X_valid[0]


n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)


#y_valid[0]


model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.summary()

#(64*784)

#(64*784)+64

#(10*64)+10

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
time0 = time()
model.fit(X_train, y_train, batch_size=128, epochs=200, verbose=1, validation_data=(X_valid, y_valid))
time1 = time()
model.evaluate(X_valid, y_valid)

print("El programa se tardo en entrenar: {:.2f} min".format((time1 - time0)/60))


