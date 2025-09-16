#Stochastic gradient descent or SGD is an improved version of gradient descent that is less likely to get stuck in local minima.
#Here, the SGD loss function value quickly falls below the losses for the more recently developed RMS Prop and the Adam optimizers on a simple minimization task. 
#Adam and RMS require 10 times as many iterations to achieve a similar loss.
import pandas as pd, numpy as np
import tensorflow as tf
import keras
import utils

tf.keras.optimizers.SGD()
#learning rate between 0.5 and 0.01  (faster may miss the minimum)

tf.keras.optimizers.RMSprop()
#2 advantages over SGD - First, it applies different learning rates to each feature, which can be useful for high dimensional problems. 
# And second, it allows you to both build momentum and also allow it to decay. 
# Setting a low value for the decay parameter will prevent momentum from accumulating over long periods during the training process.

tf.keras.optimizers.Adam()
#params - learning_rate and beta1 (decay)
#decay faster by lowering the beta1 parameter
#pefforms better with default parameter values