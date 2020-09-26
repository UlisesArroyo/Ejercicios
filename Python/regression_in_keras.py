import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from keras.layers.normalization import BatchNormalization
from time import time

(X_train, y_train), (X_valid, y_valid) = boston_housing.load_data()



print(X_train[0])
print(y_train[0])


model = Sequential()

model.add(Dense(32, input_dim=13, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

time1 = time()
model.fit(X_train, y_train,batch_size=8, epochs=32, verbose=1,validation_data=(X_valid, y_valid))


time2 = time()        
print("\nTraining Time (in seconds) =",(time2-time1))   
print(X_valid[42])

print(y_valid[42])	

print("X: ",X_train.shape)
print("Y: ",y_train.shape)


model.predict(np.reshape(X_valid[42], [1, 13]))

