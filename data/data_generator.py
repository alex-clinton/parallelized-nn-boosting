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
from keras.utils.vis_utils import plot_model

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize
x_train = x_train/255
x_test = x_test/255

number_of_classes = 10

# Create binary matrices based on number of classes
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)


# Adding this extra dimension is needed for the data generator
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# Create the generator with augmentation options
gen = ImageDataGenerator(width_shift_range=0.05, height_shift_range=0.05)

aug_data = []
aug_labels = []
num_augmented = 0
batch_size = 1000

for x_batch, y_batch in gen.flow(x_train, y_train, batch_size=batch_size, shuffle=False):
    # Add data to aug list
    aug_data.append(x_batch)
    aug_labels.append(y_batch)

    # Stop the process after reaching target number of images
    num_augmented += batch_size
    if num_augmented > 640000-batch_size:
        break

# Turn data into single arrays
aug_data = np.concatenate(aug_data)
aug_labels = np.concatenate(aug_labels)

# Combine the original and augmented data
comb_x_train = np.concatenate((aug_data, x_train))
comb_y_train = np.concatenate((aug_labels, y_train))

# Shuffle the combined data
shuffle_comb = np.random.permutation(len(comb_x_train))
comb_x_train = comb_x_train[shuffle_comb]
comb_y_train = comb_y_train[shuffle_comb]

# Recover the actual labels to use for evaluating predictions
comb_y_train_labels = np.where(comb_y_train == 1)[1]
comb_x_train = np.squeeze(comb_x_train)

np.save('comb_x_train.npy', comb_x_train)
np.save('comb_y_train.npy', comb_y_train)
np.save('comb_y_train_labels.npy', comb_y_train_labels)

np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

print(comb_x_train.shape)
print(comb_y_train.shape)
print(comb_y_train_labels.shape)
print(x_test.shape)
print(y_test.shape)