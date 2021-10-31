import tensorflow as tf
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from keras import layers
df_raw= pd.read_csv('/Users/aryanjha/Downloads/car_prices.csv')

# importing the dataset
df_raw= pd.read_csv('/Users/aryanjha/Downloads/car_prices.csv')
df = df_raw.dropna()
df.drop(['vin'],axis=1,inplace=True)
df.drop(['mmr'],axis=1,inplace=True)
df.drop(['saledate'],axis=1,inplace=True)
df.drop(['seller'],axis=1,inplace=True)
df.drop(['transmission'],axis=1,inplace=True)

# one-hot encoding the country column and the banking_crisis column
df = pd.concat([df, pd.get_dummies(df['make'], prefix='make')], axis=1)
df = pd.concat([df, pd.get_dummies(df['model'], prefix='model')], axis=1)
df = pd.concat([df, pd.get_dummies(df['trim'], prefix='trim')], axis=1)
df = pd.concat([df, pd.get_dummies(df['body'], prefix='body')], axis=1)
df = pd.concat([df, pd.get_dummies(df['state'], prefix='state')], axis=1)
df = pd.concat([df, pd.get_dummies(df['color'], prefix='color')], axis=1)
df = pd.concat([df, pd.get_dummies(df['interior'], prefix='interior')], axis=1)
df.drop(['make'],axis=1,inplace=True)
df.drop(['model'],axis=1,inplace=True)
df.drop(['trim'],axis=1,inplace=True)
df.drop(['body'],axis=1,inplace=True)
df.drop(['state'],axis=1,inplace=True)
df.drop(['color'],axis=1,inplace=True)
df.drop(['interior'],axis=1,inplace=True) 


# specifying data and labels
labels = df.iloc[:37316, [0, 1, 2, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
data = df.iloc[:37316,3]

# creating the model
model = Sequential()
model.add(layers.Dropout(0.2, input_shape=(21,)))
model.add(Dense(1000, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(21, activation='relu'))
# use government sites for weather

# 4 hidden layers of 1000: mse of 55542088

# Compile model
model.compile(loss='mse', optimizer='adam')
 
# Fit the model
model.fit(labels, data, epochs=10)

# test
data_test = df.iloc[37317:, 3]
label_test = df.iloc[37317:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]

predictions = model.predict(label_test)
predictions_flat = predictions.mean(axis=1)
print(predictions_flat)
print(data_test)
# defining mse function
def mse_predict(data, pred): 
    actual, pred = np.array(data), np.array(pred)
    return np.square(np.subtract(data, pred)).mean() 
mse_predict(data_test, predictions_flat)