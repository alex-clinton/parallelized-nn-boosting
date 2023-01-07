from mpi4py import MPI
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
import os
import time
from keras.utils.vis_utils import plot_model

# MPI Variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Batch and test size
batch_size = 40000
test_size = batch_size//4

# Training splits
first_split = 20
second_split = 50

# Training and testing data
comb_x_train = np.load('../data/comb_x_train.npy')
comb_y_train = np.load('../data/comb_y_train.npy')
comb_y_train_labels = np.load('../data/comb_y_train_labels.npy')

x_test = np.load('../data/x_test.npy')
y_test = np.load('../data/y_test.npy')

# Auto detect train size and create the partition
train_size = comb_x_train.shape[0]
partition = int((train_size-batch_size)/2 + batch_size)

net_layers = [128,100]
# Returns new network model instance
def create_network():
    model = keras.models.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(net_layers[0],'relu'),
        layers.Dense(net_layers[1],'relu'),
        layers.Dense(10,'softmax')
    ])
    # Compile and return model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 1st neural network
if rank == 0:
    # Check that the data partition is within bounds
    assert(partition < train_size)

    # Train first net 
    model1 = create_network()
    model1.fit(comb_x_train[:batch_size], comb_y_train[:batch_size], epochs=first_split)

    # Predict on the entire set (this won't all be needed but prevents costly reprediction)
    m1_pred = np.argmax(model1.predict(comb_x_train[batch_size:partition]), axis=1)
    m1_corr = np.where(m1_pred == comb_y_train_labels[batch_size:partition])[0]
    m1_incorr = np.where(m1_pred != comb_y_train_labels[batch_size:partition])[0]

    # Picks out correct and incorrect examples
    # Account for the offset 
    m1_corr_sampl = m1_corr[:batch_size//2] + batch_size
    m1_incorr_sampl = m1_incorr[:batch_size//2] + batch_size

    # Concatinate right and wrong examples
    batch_2_x_unshuff = np.concatenate((comb_x_train[m1_corr_sampl], comb_x_train[m1_incorr_sampl]))
    batch_2_y_unshuff = np.concatenate((comb_y_train[m1_corr_sampl], comb_y_train[m1_incorr_sampl]))

    assert(len(batch_2_x_unshuff) == batch_size)

    # Shuffle training data
    shuffle_b2 = np.random.permutation(len(batch_2_x_unshuff))
    batch_2_x = batch_2_x_unshuff[shuffle_b2]
    batch_2_y = batch_2_y_unshuff[shuffle_b2]

    # Make the second data set and save it for the next net
    try:
        os.mkdir('../data/predictions')
        os.mkdir('../data/train_set2')
    except:
        print('Directory for training set 2 already exists')

    # Save second training dataset
    np.save('../data/train_set2/batch_2_x.npy', batch_2_x)
    np.save('../data/train_set2/batch_2_y.npy', batch_2_y)

    # Eager predictions that will be used later for making dataset 3
    m1_ds3pred = np.argmax(model1.predict(comb_x_train[partition:]), axis=1)
    np.save('../data/predictions/m1_ds3pred.npy', m1_ds3pred)

    # Alert 2nd process that model1 has been saved
    comm.send('\n Model1 passed on\n', dest=1, tag=0)

    # Finish training on first net
    model1.fit(comb_x_train[:batch_size], comb_y_train[:batch_size], epochs=second_split)

    # Used the fully trained model1 to make final predictions on the test data
    m1_final_pred = model1.predict(x_test[:test_size])
    np.save('../data/predictions/m1_final_pred.npy', m1_final_pred)

elif rank == 1:
    # Wait for the signal from process 1 to start 
    message = comm.recv(tag=0, source=0)
    print(message)

    batch_2_x = np.load('../data/train_set2/batch_2_x.npy')
    batch_2_y = np.load('../data/train_set2/batch_2_y.npy')

    # Create second network
    model2 = create_network()
    model2.fit(batch_2_x, batch_2_y, epochs=first_split)

    # Make training set 3
    # Predictions for first 2 networks
    # Loaded from another file so that no time is wasted loading the models
    m1_ds3pred = np.load('../data/predictions/m1_ds3pred.npy')
    m2_pred = np.argmax(model2.predict(comb_x_train[partition:]), axis=1)
    disagreements = np.where(m1_ds3pred != m2_pred)[0] + 10*batch_size

    # Make sure that our data set size is actually big enough
    assert(len(disagreements[:batch_size]) == batch_size)

    # Third batch
    batch_3_x = comb_x_train[disagreements[:batch_size]]
    batch_3_y = comb_y_train[disagreements[:batch_size]]

    try:
        os.mkdir('../data/train_set3')
    except:
        print('Directory for training set 3 already exists')

    np.save('../data/train_set3/batch_3_x', batch_3_x)
    np.save('../data/train_set3/batch_3_y', batch_3_y)

    # Alert 2nd process that model1 has been saved
    comm.send('\n Model2 passed on\n', dest=2, tag=0)

    # Finish training model 2
    model2.fit(batch_2_x, batch_2_y, epochs=second_split)

    # Used the fully trained model 2 to make final predictions on the test data
    m2_final_pred = model2.predict(x_test[:test_size])
    np.save('../data/predictions/m2_final_pred.npy', m2_final_pred)

elif rank == 2:
    start = time.time()

    # Wait for the single from process 2 to start 
    message = comm.recv(tag=0, source=1)
    print(message)

    # Third batch
    batch_3_x = np.load('../data/train_set3/batch_3_x.npy')
    batch_3_y = np.load('../data/train_set3/batch_3_y.npy')

    model3 = create_network()
    total_split = first_split + second_split
    model3.fit(batch_3_x, batch_3_y, epochs=total_split)

    end = time.time()
    print('Time:', str(end-start))

    # Individual Predictions (loading other two pre-predictions)
    m1_final_pred = np.load('../data/predictions/m1_final_pred.npy')
    m2_final_pred = np.load('../data/predictions/m2_final_pred.npy')
    m3_final_pred = model3.predict(x_test[:test_size])

    # Aggregate Predicitons
    total_pred = m1_final_pred + m2_final_pred + m3_final_pred

    final_pred = np.argmax(total_pred, axis=1)
    m1_final_pred_class = np.argmax(m1_final_pred, axis=1) 
    m2_final_pred_class = np.argmax(m2_final_pred, axis=1) 
    m3_final_pred_class = np.argmax(m3_final_pred, axis=1) 

    # Calculate Accuracy
    y_test_labels = np.where(y_test[:test_size] == 1)[1]

    wrong_pred = np.where(final_pred != y_test_labels[:test_size])[0]
    m1_wrong_pred = np.where(m1_final_pred_class != y_test_labels[:test_size])[0]
    m2_wrong_pred = np.where(m2_final_pred_class != y_test_labels[:test_size])[0]
    m3_wrong_pred = np.where(m3_final_pred_class != y_test_labels[:test_size])[0]

    accuracy = (test_size-len(wrong_pred))/test_size
    m1_accuracy = (test_size-len(m1_wrong_pred))/test_size
    m2_accuracy = (test_size-len(m2_wrong_pred))/test_size
    m3_accuracy = (test_size-len(m3_wrong_pred))/test_size

    print('Accuracy:', accuracy)

    # Write results to file
    with open('parallel_boosting/parallel_boosting_results.txt', 'a') as f:
        f.write(f'FMNIST\t accuracy: {accuracy:.3f}\t time: {end-start:.2f}\t split: {first_split}/{second_split}\n')
        # Individual evalutaions
        f.write(f'm1 acc: {m1_accuracy:.3f}\t m2 acc: {m2_accuracy:.3f}\t m3 acc: {m3_accuracy:.3f}\n')
        # Network architecture and training data stats
        f.write(f'layer nodes: {net_layers}\t batch size: {batch_size}\t test size: {test_size}\n')
        f.write('\n')