import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
#CIFAR 10 

(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data(label_mode='fine')
#CIFAR 100 