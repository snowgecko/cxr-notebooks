####work on Tensorflow 
import pandas as pd, numpy as np
import tensorflow as tf
import keras
import utils

keras.config.set_backend("tensorflow")  # or set env: KERAS_BACKEND=tensorflow
print("Keras:", keras.__version__)

#define intercept and slope 
intercept = tf.Variable(0.1, np.float32)
slope = tf.Variable(0.1, np.float32)

#housing = pd.read_csv('kc_housing.csv')
for batch in pd.read_csv('kc_housing.csv', chunksize=100):
    price_batch = np.array(batch['price'], np.float32)
    size_batch = np.array(batch['size'], np.float32)
    #waterfront = np.array(batch['waterfront'], bool)
    
    #define - wil lower the value of the loss
    opt = tf.keras.optimizers.Adam()
    opt.minimize(lambda: utils.loss_function(intercept, slope, price_batch, size_batch), \
                 var_list = [intercept, slope])


#passing a loss function as a lambda in this context is a clever way to wrap the computation so it can be dynamically evaluated during optimization.
# minnimise loss function and print the loss
#for j in range(1000):
#    opt.minimize(lambda: utils.loss_function(intercept, slope), \
#                 var_list = [intercept, slope])
#    print(utils.loss_function(intercept,slope))

#or we can use the cast operation from tensorflow 
#price = tf.cast(housing['price'],tf.float32)
#price = tf.cast(housing['waterfront'],tf.bool)

#Loss functions used to train models - Hiher value = worse fit --> minimise the loss function
#MSE - mean squared error loss - strongly penalises outliers - high sensitivity near gradient 
#MAE - mean absolute error - scales linearly witht he size of the error - low sensitivity near gradient 
#Huber error - similar to MSE - near 0, similar to MAE away from 0 

#loss = tf.keras.losses.mse(targets, predictions)

# Initialize a variable named scalar
#scalar = tf.Variable(1.0, )

# Evaluate the loss function and print the loss
#print(utils.loss_function_mae(scalar).numpy())

# Linear regression assumes linear relationship after taking the natural log of x % increase in size with y% increase in size 
# assunmmed linear relationahip - after taking natural log




