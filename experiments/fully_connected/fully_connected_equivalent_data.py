from datetime import datetime
from packaging import version
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import tensorboard
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import time
import os
from keras.utils.vis_utils import plot_model

start = time.time()

comb_x_train = np.load('../data/comb_x_train.npy')
comb_y_train = np.load('../data/comb_y_train.npy')

x_test = np.load('../data/x_test.npy')
y_test = np.load('../data/y_test.npy')

batch_size = 120000
train_epoch = 40
start = time.time()

net_layers = [384,300]

model = keras.models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    # 128 and 100 x 3
    layers.Dense(net_layers[0],'relu'),
    layers.Dense(net_layers[1],'relu'),
    layers.Dense(10,'softmax')
])

# Compile and return model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(comb_x_train[:batch_size], comb_y_train[:batch_size], epochs=train_epoch)

end = time.time()

results = model.evaluate(x_test, y_test)

with open('fully_connected/fully_connected_results_equivalent_data.txt', 'a') as f:
    f.write(f'FMNIST\t acc: {results[1]:.4f}\t time: {end-start:.2f}\t arch: {net_layers}\t epochs: {train_epoch}\t batch size: {batch_size}\n')
