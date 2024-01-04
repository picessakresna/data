import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import kaggle

# Set the Kaggle API credentials
kaggle.api.authenticate()

# Download the dataset
kaggle.api.dataset_download_files('berkeleyearth/climate-change-earth-surface-temperature-data', path='./dataset', unzip=True)

files = os.listdir('./dataset')

for file in files:
    if file != 'GlobalLandTemperaturesByMajorCity.csv':
        os.remove(os.path.join('./dataset', file))

df = pd.read_csv('./dataset/GlobalLandTemperaturesByMajorCity.csv')
df.head

df.drop(['City', 'Latitude', 'Longitude'], axis = 1, inplace = True)
print(df)

df['dt'] = pd.to_datetime(df['dt'])
get_data = (df['dt'] > '1945-01-01') & (df['dt'] <= '2013-09-01')
df.loc[get_data]

df = df.loc[get_data]
print(df)

df = df.loc[df['Country'].isin(['China'])]
print(df)

df.drop(['Country'], axis = 1, inplace = True)
df.reset_index(drop = True)

df.isnull().sum()

df.dropna(subset = ['AverageTemperature'], inplace = True)
df.dropna(subset = ['AverageTemperatureUncertainty'], inplace = True)
df.isnull().sum()

df_plot = df
df_plot[df_plot.columns.to_list()].plot(subplots = True, figsize = (20, 10))
plt.show()

dates = df['dt'].values
temp = df['AverageTemperature'].values

dates = np.array(dates)
temp = np.array(temp)

plt.figure(figsize = (20, 10))
plt.plot(dates, temp)

plt.title('Average Temperature', fontsize = 15)
plt.ylabel('Temperature')
plt.xlabel('Datetime')

# split dataset
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(temp, dates, train_size = 0.8, test_size = 0.2, shuffle = False)

print('Jumlah Data Train : ', len(x_train))
print('Jumlah Data Validation : ', len(x_valid))

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  series = tf.expand_dims(series, axis = -1)
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size + 1, shift = 1, drop_remainder = True)
  ds = ds.flat_map(lambda w : w.batch(window_size + 1))
  ds = ds.shuffle(shuffle_buffer)
  ds = ds.map(lambda w : (w[:-1], w[-1:]))
  return ds.batch(batch_size).prefetch(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Bidirectional,Dropout

tf.keras.backend.set_floatx('float64')

train_set = windowed_dataset(x_train, window_size = 64, batch_size = 200, shuffle_buffer = 1000)
val_set = windowed_dataset(x_valid, window_size = 64, batch_size = 200, shuffle_buffer = 1000)

model = Sequential([
    Bidirectional(LSTM(60, return_sequences=True)),
    Bidirectional(LSTM(60)),
    Dense(30, activation="relu"),
    Dense(10, activation="relu"),
    Dense(1),
])

Mae = (df['AverageTemperature'].max() - df['AverageTemperature'].min()) * 10/100
print(Mae)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('mae') < 5.5 and logs.get('val_mae') < 5.5):
      print("Mae < 10% data")
      self.model.stop_training = True
callbacks = myCallback()

optimizer = tf.keras.optimizers.SGD(learning_rate=1.0000e-04, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

history = model.fit(train_set, epochs=100, validation_data = val_set, callbacks=[callbacks])

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('Mae')
plt.legend(['Train', 'Val'], loc = 'best')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train', 'Val'], loc = 'best')
plt.show()