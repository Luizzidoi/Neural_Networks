"""
Convolutional Neural Networks
MNIST Data Base -> Digits
1 - Analyze neural network precision
2 - Buscar um número aleatório da base MNIST e classificar o digíto correspondente
2 - Search a random number of the MNIST data base and classify corresponding digit
"""

""" ----------------------------- Import libraries ----------------------------- """
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
# Flatten -> Tirth stage of CNN
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
# Conv2D is first stage (convolution operator) and MaxPooling2D is second stage (pooling) of CNN
from keras.layers import Conv2D, MaxPooling2D
# Function used to features maps normalization. Neural Networks improvements
from tensorflow.keras.layers import BatchNormalization
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



""" --------- Load database in variables: x (forecasters) and y (classes) --------- """
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Search a image of database. cmap = 'gray' -> show a image without color to decrease the processing
plt.imshow(X_train[0], cmap='gray')
plt.title('Class ' + str(y_train[0]))

## Image preview
# plt.show()


""" ------ Transformation on dataset so that the Tensorflow get to do the read ------ """
# reshape = change data format
prev_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
print('X_train.shape[0] means the position 0 of (60000, 28, 28):',  X_train.shape[0])
prev_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# astype = change variable type to uint32
prev_train = prev_train.astype('float32')
prev_test = prev_test.astype('float32')

# Modify values scale to processing be faster. Transform scale 0 to 255 for 0 to 1.
# Technique to min/max normalization = dataset normalization to decrease scale
prev_train /= 255
prev_test /= 255

# Modify the classes to Dummy type
class_train = np_utils.to_categorical(y_train, 10)
class_test = np_utils.to_categorical(y_test, 10)


""" ---------------------------- Neural Networks structure ---------------------------- """
# Stage 1 = Convolution operator
classificator = Sequential()
classificator.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
# Only to improvements of neural network. Feature map normalization
classificator.add(BatchNormalization())
# Stage 2 = Pooling
classificator.add(MaxPooling2D(pool_size=(2, 2)))
# Stage 3 = Flattening
# classificador.add(Flatten())

# Add one more convolution layer to results improvements
classificator.add(Conv2D(32, (3, 3), activation='relu'))
# Only to neural network improvements, feature map normalization
classificator.add(BatchNormalization())
classificator.add(MaxPooling2D(pool_size=(2, 2)))
# Stage 3 - Flattening
# When add one more convolution layer, the Flattening need to be implemented only once in the end
classificator.add(Flatten())

# Stage 4 = Dense neural network creation 
classificator.add(Dense(units=128, activation='relu'))
classificator.add(Dropout(0.2))
classificator.add(Dense(units=128, activation='relu'))
classificator.add(Dropout(0.2))
classificator.add(Dense(units=10, activation='softmax'))
classificator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# validation_data = neural network training and make the tests with 'forecasters_test' and 'classes_test' datasets
classificator.fit(prev_train, class_train, batch_size=128, epochs=10,
                  validation_data=(prev_test, class_test))

""" Print neural network accuracy - it's same use the 'validation_data' parameter previously  """
results = classificator.evaluate(prev_test, class_test)
print(results)


""" ----------------------------- Prevision of one image ----------------------------- """
# Using a image of test dataset with value 7
plt.imshow(X_test[0], cmap='gray')
plt.title('Class' + str(y_test[0]))
plt.show()

# Variable that storage a image to be classified and
# Needed to do transformation on dimension to processing Tensorflow
image_test = X_test[0].reshape(1, 28, 28, 1)
image_test = image_test.astype('float32')
image_test /= 255

# Como temos um problema multiclasse e a função de ativação softmax, será gerada uma probabilidade para
# cada uma das classes. A variável previsão terá a dimensão 1x10, sendo que em cada coluna estará o
# valor de probabilidade de cada classe
prevision = classificator.predict(image_test)
print(prevision)

# How each vector index represents a number between 0 and 9, now we seek what the bigger index and return it. 
# Executing code below we have the index 7 that represents class 7
import numpy as np
number = np.argmax(prevision)
print('The number is:', number)


""" ----------------------- Saving the neural network in JSON format ----------------------- """
# Save as string all data that was passed to neural network (neural networks structure)
classificator_json = classificator.to_json()

# Saving the neural network structure in disk
with open('classificador_RNC_MNIST.json', 'w') as json_file:
    json_file.write(classificator_json)

# Saving the neural network weights
classificator.save_weights('classificator_RNC_MNIST.h5')
