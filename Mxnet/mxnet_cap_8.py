#version buena
import mxnet as mx
import logging



#------------------------CAPTURA Y PROCESAMIENTO DE BASE DE DATOS-------------------------------
mnist = mx.test_utils.get_mnist() #Descarga y carga las imagenes y las etiquetas en la memoria


batch_size = 128				#Tamaño de los lotes
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)#La base de datos se va a dividir de manera aleatoria en lotes de 128 tanto las imagenes como las y's
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)					#Lo mismo solo que se seccionara de manera ordenada
#				ITERADORES


data = mx.sym.var('data') #Crea la variable que sera modifcada pero no se si carga la informacion de train_iter , val_iter o de los 2

# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
data = mx.sym.flatten(data=data) #Aplana nuestras imagenes de 28 * 28 a 784

#------------------------CAPTURA Y PROCESAMIENTO DE BASE DE DATOS-------------------------------




#------------------------CONSTRUCCION DE RED NEURONAL-------------------------------

# The first fully-connected layer and the corresponding activation function
fc1  = mx.sym.FullyConnected(data=data, num_hidden=64) #Se crea la capa densa y se especifica el numero de neuronas que tendra. Tambien especifica que informacion se le introducira
act1 = mx.sym.Activation(data=fc1, act_type="relu")		#Se especifica el tipo de funcion de activacion que tendra la primera capa

# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")


# MNIST has 10 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)	#Ultima capa 
# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')	#La ultima funcion se especifica, en este caso softmax

#------------------------CONSTRUCCION DE RED NEURONAL-------------------------------


logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on CPU
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
mlp_model.fit(train_iter,  # train data // Info de entrenamiento
              eval_data=val_iter,  # validation data // Info de validacion
              optimizer='sgd',  # use SGD to train // Tipo de optimizador a utilizar
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate // La taza de aprendizaje 
              eval_metric='acc',  # report accuracy during training // reporte de exactitud durante el entrenamiento
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches // progreso por cada 100 lotes
              num_epoch=20)  # train for at most 10 dataset passes // cantidad de epocas 


#Realiza una prediccion a la probabilidad que tendra la red neuronal en determinar el numero de la imagen correctamente
test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size) 
prob = mlp_model.predict(test_iter)
assert prob.shape == (10000, 10)

#Se calcula la precision que tuvo la red neuronal 
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than expected (0.96)" % acc.get()[1]

