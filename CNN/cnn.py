# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 22:29:19 2018

@author: konyd
"""

# You can initialize a neural network either as a sequence of
# layers or as a graph. Here we want a sequence
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import h5py

# Initialize the CNN
classifier = Sequential()

# Define the convolutional layer. Create 32 feature
# detectors for a 32 feature maps. Activation function to make sure
# no negative values are present in the image.
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Pooling. Do on the max-pooling basis
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add a second convolutional layer
# It's a common strategy to double the number of feature detectors. 
classifier.add(Convolution2D(20, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
classifier.add(Flatten())

# The FC-ANN. The hidden layer with added 60% dropout. 
classifier.add(Dense(128, activation='relu', use_bias=True))
classifier.add(Dropout(rate=0.75))

# A second hidden layer
classifier.add(Dense(64, activation='relu', use_bias=True))
classifier.add(Dropout(rate=0.8))

# Output layer. Sigmoid, because it's a squashing function between
# 0 and 1. Perfect for binary predictions like the current one.
classifier.add(Dense(1, activation='sigmoid'))

# Compile the model. Binary_crossentropy is best used for binary output predictions,
# categorycal crossentropy - for multiple prediction classes.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation. As the dataset is not too big, we will introduce
# some 'mutations', random transformations, per batch of images we
# train on the model. A technique to avoid overfitting.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=64,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, # Number of images in the training set
        epochs=26,
        validation_data=test_set,
        validation_steps=2000) # Number images of out test set.

# Save only the weights of the model
classifier.save_weights('weightsCatDogCNN.hdf5')