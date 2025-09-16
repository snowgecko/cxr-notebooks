
import tensorflow as tf

def linear_regression(intercept, slope = slope, features = size):
    return intercept + features*slope


def loss_function(intercept, slope, targets, features):
    predictions = linear_regression(intercept, slope)
    #return the loss
    return tf.keras.losses.mse(targets, predictions)

# Define a loss function
def loss_function_ps(intercept, slope, targets = price, features = size):
    predictions = linear_regression(intercept, slope)
    #return the loss
    return tf.keras.losses.mse(targets, predictions)


# Define a loss function
def loss_function_mse(intercept, slope, targets = targets, features = features):
    predictions = linear_regression(intercept, slope)
    #return the loss
    return tf.keras.losses.mse(targets, predictions)

# Define a loss function
def loss_function_mae(scalar, targets = targets, features = features):
	# Compute the predicted values
	predictions = model(scalar, features)    
	# Return the mean absolute error loss
	return tf.keras.losses.mae(targets, predictions)


# Define the model
def model(scalar, features = features):
  	return scalar * features


