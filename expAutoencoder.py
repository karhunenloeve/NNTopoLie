#!/usr/bin/env python

from autoencoderInvertible import *
from keras.models import Model
from keras.losses import kullback_leibler_divergence, mean_squared_error
from keras.optimizers import Adam
from keras.datasets import mnist, cifar10, cifar100, boston_housing
from keras.callbacks import TensorBoard, CSVLogger
from multiprocessing import Pool

from os import listdir
from os.path import isfile, join


def list_diff(list1, list2): 
    return (list(set(list1) - set(list2))) 

def execute_experiments(i):
    print("Current dimension: " + str(i))
    invertible_autoencoder = Model(
    input_img, dense_invertible_subspace_autoencoder(input_img, units=i, invertibleLayers=5, groupLayers=3)
    )
    invertible_autoencoder.summary()
    invertible_autoencoder.compile(optimizer=Adam(lr=0.0001), loss=mean_squared_error)
    csv_logger = CSVLogger('results/cifar100/c_1_log_units'+ str(i) + '.csv', append=True, separator=',')
    invertible_autoencoder.fit(
        add_gaussian_noise(x_train, 0.8),
        x_train,
        epochs=250,
        batch_size=128,
        shuffle=True,
        validation_data=(add_gaussian_noise(x_test, 0.8), x_test),
        callbacks = [csv_logger],
        verbose=0
    )    

def homgroups_dimension():
    cifar10 = [12,16,40,59,50]
    cifar100 = [13,18,34,46,48]

    epsilon = 30.

    for homologygroup in range(1,5):
        homcount = cifar100[homologygroup]
        for p in range(3, 1000):
            number = 1.
            for i in range(1, homologygroup + 1):
                number = number * ((p - homologygroup + i) / i)
            if number > (homcount - epsilon) and number < (homcount + epsilon):
                print("The amount of needed spheres for H" + str(homologygroup) + " is approximately " + str(p) + " for epsilon = " + str(number - homcount))

homgroups_dimension()