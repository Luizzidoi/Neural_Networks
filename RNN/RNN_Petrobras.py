"""
Recurrent Neural Networks
Base de dados Bolsa de valores
Stock exchange database
Stock Price prediction
Generating a graph to compare the real price with the one predicted by the neural network
"""


import pandas as pd
from keras.models import Sequential
# LSTM - neural network type used, one more of efficient
from keras.layers import Dense, Dropout, LSTM
# Normalizing values ​​between 0 and 1
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" -------------------- Loading variables with database attributes -------------------- """
base = pd.read_csv('petr4_treinamento.csv')
# Removing NAN values ​​from the database
base = base.dropna()
# Using the database's "Open" parameter for training
base_train = base.iloc[:, 1:2].values

# Normalize values ​​between 0 and 1. Minimize processing
normalizator = MinMaxScaler(feature_range=(0, 1))
base_train_normalized= normalizator.fit_transform(base_train)


""" --------- Filling in the variables with 90 previous dates for the training --------- """
predictors = []
real_value = []
# 90 previous values ​​for predictors and 1242 the size of the database
for i in range(90, 1242):
    predictors.append(base_train_normalized[i-90:i, 0])
    real_value.append(base_train_normalized[i, 0])

# Transforming data for a table
predictors, real_value = np.array(predictors), np.array(real_value)

# Transforming the data so that Tensorflow can read it. Input shape (batch_size, timesteps, input_dim)
predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1))


""" ------------------------ Recurrent Neural Networks structure ------------------------ """
# units = number of memory cells will be added in the layer
# return_sequences = used when there is more than one LSTM layer. Passes data to subsequent layers
regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(predictors.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='linear'))

# 'rsmprop' optimizer - the most used for recurrent neural networks
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

regressor.fit(predictors, real_value, epochs=150, batch_size=32)



""" ------------------------ Saving the neural network to disk ------------------------ """

# Saves as a String all the data that was passed to the network (Neural network structure)
regressor_json = regressor.to_json()

# Saving the neural network to disk
with open('regressor_1Previsor_1Saida.json', 'w') as json_file:
    json_file.write(regressor_json)

# Saving Neural Network Weights
regressor.save_weights('regressor_1Prev_1output.h5')



""" --------------------- Using the test database for better results ---------------------"""
base_test = pd.read_csv('petr4_teste.csv')
real_value_test = base_test.iloc[:, 1:2].values
# Concatenate the training database with the test database to be able to search/use the previous 90 values.
# Axis=0 for concatention by column
full_base = pd.concat((base['Open'], base_test['Open']), axis=0)
# Search 90 previous values
inputs = full_base[len(full_base) - len(base_test) - 90:].values
# Leave it in numpy format: (112, 1) -> '-1' when you don't want to work with the lines
inputs = inputs.reshape(-1, 1)
# Normalization of inputs
inputs = normalizator.transform(inputs)



""" -------------------------------- Using the test base -------------------------------- """
X_test = []
# 90 is the start of the test base and 112 is the end
for i in range(90, 112):
    X_test.append(inputs[i-90:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred = regressor.predict(X_test)
pred = normalizator.inverse_transform(pred)

diff = pred.mean() - real_value_test.mean()
print("Difference mean between real value and predictions: ", diff)


""" ------------------------- Generating the graph for analysis ------------------------- """
plt.plot(real_value_test, color='red', label='Real value')
plt.plot(pred, color='blue', label='Predictions')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()