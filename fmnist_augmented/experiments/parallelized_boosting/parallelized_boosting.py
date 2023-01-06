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

'''
Things to keep in mind/check while running tests
1. # of epochs and epoch split
2. training set size
3. file that is being written to
4. network architecture
5. dataset
'''

#######################################
# Variables across all script instances
#######################################

# MPI Variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Batch and test size
bs = 40000
ts = int(bs*.25)

# Training splits
first_split = 20
second_split = 50

# How to split training data to make the multiple datasets
'''
bs               part
--------------------------------------------
|  ds1  |       ds2      |        ds3      |
--------------------------------------------
'''

# Training and testing data
comb_x_train = np.load('../data/comb_x_train.npy')
comb_y_train = np.load('../data/comb_y_train.npy')
comb_y_train_labels = np.load('../data/comb_y_train_labels.npy')

x_test = np.load('../data/x_test.npy')
y_test = np.load('../data/y_test.npy')

# Auto detect train size and create the partition
train_size = comb_x_train.shape[0]
part = int((train_size-bs)/2 + bs)

################################################
# Network definition across all script instances
################################################

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

#####################################
# Cases depending on taining instance
#####################################

# 1st neural network
if rank == 0:

    # Check that the data partition doesn't go out of bounds
    assert(part < train_size)

    # Train first net 
    model1 = create_network()
    model1.fit(comb_x_train[:bs], comb_y_train[:bs], epochs=first_split)

    # Predict on the entire set (this won't all be needed but prevents costly reprediction), *10 is arbitrary
    m1_pred = np.argmax(model1.predict(comb_x_train[bs:part]), axis=1)
    m1_corr = np.where(m1_pred == comb_y_train_labels[bs:part])[0]
    m1_incorr = np.where(m1_pred != comb_y_train_labels[bs:part])[0]

    # Picks out correct and incorrect examples
    # Account for the offset 
    m1_corr_sampl = m1_corr[:int(bs/2)] + bs
    m1_incorr_sampl = m1_incorr[:int(bs/2)] + bs

    # Concatinate right and wrong examples
    batch_2_x_unshuff = np.concatenate((comb_x_train[m1_corr_sampl], comb_x_train[m1_incorr_sampl]))
    batch_2_y_unshuff = np.concatenate((comb_y_train[m1_corr_sampl], comb_y_train[m1_incorr_sampl]))

    assert(len(batch_2_x_unshuff) == bs)

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

    # save second training dataset
    np.save('../data/train_set2/batch_2_x.npy', batch_2_x)
    np.save('../data/train_set2/batch_2_y.npy', batch_2_y)

    # Eager predictions that will be used later for making dataset 3
    m1_ds3pred = np.argmax(model1.predict(comb_x_train[part:]), axis=1)
    np.save('../data/predictions/m1_ds3pred.npy', m1_ds3pred)

    # Alert 2nd process that model1 has been saved
    comm.send('\n Model1 passed on\n', dest=1, tag=0)

    # Finish training on first net
    model1.fit(comb_x_train[:bs], comb_y_train[:bs], epochs=second_split)

    # Used the fully trained model1 to make final predictions on the test data
    m1_final_pred = model1.predict(x_test[:ts])
    np.save('../data/predictions/m1_final_pred.npy', m1_final_pred)

elif rank == 1:

    # Wait for the single from process 1 to start 
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
    m2_pred = np.argmax(model2.predict(comb_x_train[part:]), axis=1)
    disagreements = np.where(m1_ds3pred != m2_pred)[0] + 10*bs

    # Make sure that our data set size is actually big enough
    assert(len(disagreements[:bs]) == bs)

    # Third batch
    batch_3_x = comb_x_train[disagreements[:bs]]
    batch_3_y = comb_y_train[disagreements[:bs]]

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
    m2_final_pred = model2.predict(x_test[:ts])
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

    # Evaluation

    # Individual Predictions (loading other two pre-predictions)
    m1_final_pred = np.load('../data/predictions/m1_final_pred.npy')
    m2_final_pred = np.load('../data/predictions/m2_final_pred.npy')
    m3_final_pred = model3.predict(x_test[:ts])

    # Aggregate Predicitons
    total_pred = m1_final_pred + m2_final_pred + m3_final_pred

    final_pred = np.argmax(total_pred, axis=1)
    m1_final_pred_class = np.argmax(m1_final_pred, axis=1) 
    m2_final_pred_class = np.argmax(m2_final_pred, axis=1) 
    m3_final_pred_class = np.argmax(m3_final_pred, axis=1) 

    # Calculate Accuracy
    y_test_labels = np.where(y_test[:ts] == 1)[1]
    print(y_test_labels.shape)
    print(m1_final_pred.shape)

    wrong_pred = np.where(final_pred != y_test_labels[:ts])[0]
    m1_wrong_pred = np.where(m1_final_pred_class != y_test_labels[:ts])[0]
    m2_wrong_pred = np.where(m2_final_pred_class != y_test_labels[:ts])[0]
    m3_wrong_pred = np.where(m3_final_pred_class != y_test_labels[:ts])[0]

    accuracy = (ts-len(wrong_pred))/ts
    m1_accuracy = (ts-len(m1_wrong_pred))/ts
    m2_accuracy = (ts-len(m2_wrong_pred))/ts
    m3_accuracy = (ts-len(m3_wrong_pred))/ts

    print('Accuracy:', accuracy)

    # Write results to results.txt
    with open('parallel_boosting/parallel_boosting_results.txt', 'a') as f:
        f.write(f'FMNIST\t accuracy: {accuracy:.3f}\t time: {end-start:.2f}\t split: {first_split}/{second_split}\n')
        # Optional: individual evalutaions
        f.write(f'm1 acc: {m1_accuracy:.3f}\t m2 acc: {m2_accuracy:.3f}\t m3 acc: {m3_accuracy:.3f}\n')
        # Optional: network architecture and training data stats
        f.write(f'layer nodes: {net_layers}\t batch size: {bs}\t test size: {ts}\n')
        f.write('\n')
