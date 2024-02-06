

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.constraints import max_norm
import datetime
from PIL import Image

import os
import kaggle

dataset_folder = './dataset/Animals-10'
if not os.path.exists(dataset_folder):
    kaggle.api.dataset_download_files('viratkothari/animal10', path='./dataset', unzip=True)

os.listdir('./dataset/Animals-10')

classes_animal = {'dog': 'dog',
 'butterfly': 'butterfly',
 'sheep': 'sheep',
 'horse': 'horse',
 'spider': 'spider',
 'elephant': 'elephant',
 'chicken': 'chicken',
 'cow': 'cow',
 'cat': 'cat',
 'squirrel': 'squirrel'}

for key, value in classes_animal.items():
    source_path = f"D:/SELF/DATA/data-portofolio/image-classification/dataset/Animals-10/{value}"
    print(f"There are {len(os.listdir(source_path))} images of {value}")

# Specify the path of the dataset folder
dataset_folder = './dataset'

# Create the data folder inside the dataset folder
data_folder = os.path.join(dataset_folder, 'data')
os.makedirs(data_folder, exist_ok=True)

import os
import shutil

# Specify the source and destination folders
source_folder = './dataset/Animals-10'
destination_folder = './dataset/data'

# List all animal folders
animal_folders = ['dog', 'spider', 'chicken']

# Copy files from each animal folder to the data folder with separate folders
for animal_folder in animal_folders:
    animal_folder_path = os.path.join(source_folder, animal_folder)
    if os.path.isdir(animal_folder_path):
        destination_animal_folder = os.path.join(destination_folder, animal_folder)
        os.makedirs(destination_animal_folder, exist_ok=True)
        files = os.listdir(animal_folder_path)
        count = 0
        for file in files:
            if count >= 3500:
                break
            source_file = os.path.join(animal_folder_path, file)
            destination_file = os.path.join(destination_animal_folder, file)
            shutil.copy(source_file, destination_file)
            count += 1

data_animal = {
 'dog': 'dog',
 'spider': 'spider',
 'chicken': 'chicken'}

for key, value in data_animal.items():
    source_path = f"D:/SELF/DATA/data-portofolio/image-classification/dataset/data/{value}"
    print(f"There are {len(os.listdir(source_path))} images of {value}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR="D:/SELF/DATA/data-portofolio/image-classification/dataset/data/"
VALIDATION_DIR="D:/SELF/DATA/data-portofolio/image-classification/dataset/data/"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode = 'nearest',
                                   validation_split=0.2)

train_generator= train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training')

num_train_data = train_generator.samples
print("Train dataset:", num_train_data)

validation_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    class_mode='categorical',
    subset='validation')

num_val_data = validation_generator.samples
print("Validation dataset:", num_val_data)

labels = {value: key for key, value in train_generator.class_indices.items()}

print("Label Mappings for classes present in the training and validation datasets\n")
for key, value in labels.items():
    print(f"{key} : {value}")

import random

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

# Get a random sample of 3 classes
random_classes = random.sample(list(labels.values()), 3)

for i, value in enumerate(random_classes):
    # Get a random image from the class folder
    image_files = os.listdir(os.path.join(TRAINING_DIR, value))
    random_image = random.choice(image_files)
    image_path = os.path.join(TRAINING_DIR, value, random_image)

    # Load and plot the image
    image = plt.imread(image_path)
    ax[i].imshow(image)
    ax[i].set_title(value)
    ax[i].axis('off')

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Get a batch of augmented images and labels
images, labels = next(train_generator)

# Plot the images
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.set_title(labels[i])
    ax.axis('off')

# Show the plot
plt.tight_layout()
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if((logs.get('accuracy')>0.92 and logs.get('val_accuracy')>0.92)):
      print("\nAkurasi dan Val Akurasi telah sesuai tarhet")
      self.model.stop_training = True
callbacks_stop = myCallback()

result = model.fit(train_generator,
                       epochs = 150,
                       validation_data=(validation_generator),
                       callbacks=[callbacks_stop],
                       batch_size=128,
                       verbose=1)

plt.plot(result.history['loss'], 'black', linewidth=2.0)
plt.plot(result.history['val_loss'], 'red', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14, loc='best')
plt.title('Loss Curves', fontsize=12)
plt.ylabel('Loss', fontsize=10)
plt.xlabel('Epochs', fontsize=10)
plt.show()

plt.plot(result.history['accuracy'], 'blue', linewidth=2.0)
plt.plot(result.history['val_accuracy'], 'orange', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14, loc='best')
plt.title('Accuracy Curves', fontsize=12)
plt.ylabel('Accuracy', fontsize=10)
plt.xlabel('Epochs', fontsize=10)
plt.show()

best_accuracy = max(result.history['accuracy'])
best_val_accuracy = max(result.history['val_accuracy'])

print("Best Accuracy:", best_accuracy)
print("Best Validation Accuracy:", best_val_accuracy)

saving_path = ("./mymodel/") #path penyimpanan model
tf.saved_model.save(model, saving_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('CNN_model.tflite', 'wb') as f:
  f.write(tflite_model)