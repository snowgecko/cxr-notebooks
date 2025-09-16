####work on Tensorflow 
import pandas as pd, numpy as np
import tensorflow as tf
import keras
import utils

keras.config.set_backend("tensorflow")  # or set env: KERAS_BACKEND=tensorflow
print("Keras:", keras.__version__)

####Low level approach - linear algabraic approach
inputs = tf.constant([1,35])
weights = tf.Variable([-0.05],[-0.01])
bias = tf.Variable([0.5])
####bias smilar to intercept 
product = tf.matmul (inputs, weights)
dense = tf.keras.activations.sigmoid(product+bias)


####high level approach  - things like bias tend to be hidden
inputs = tf.constant(data, tf.float32)
dense1 = tf.keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)
# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)
# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)
# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)

#Activation functions - ie, when the ouptut is dependent on something like age - ie, a non-linear factor
# But what if we apply a sigmoid activation function? The impact of bill amount on default now depends on the borrower's age
# The sigmoid activation function is used primarily in the output layer of binary classification problems.

#Rectified linear unit or relu activation - in all layers other than the output 
#maximum of the value passed to it and 0 

#Softmax - on the output layer and those with > 2 classess

# Define the first dense layer
inputsR = tf.constant(borrower_features, tf.float32)
dense1 = keras.layers.Dense(16, activation='relu')(inputsR)
dense2 = keras.layers.Dense(8, activation='simoid')(dense1)
outputs = keras.layers.Dense(4, activation='softmax')(dense2) #since there are more thnan two outputs

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', outputs.shape)
