'''
Make a convolutional neural network (CNN) classifier for classifying handwritten digits
given in the MNIST database (http://yann.lecun.com/exdb/mnist/) that is also with this
assignment. Comment your code well explaining each step and report classifying error /
accuracy. Visualize examples of possible miss-classification. Submit a report and program
code(s).
'''
# https://www.kaggle.com/code/heeraldedhia/mnist-classifier-first-deep-learning-project
# https://www.kaggle.com/code/gpreda/simple-introduction-to-cnn-for-mnist-99-37
# https://www.geeksforgeeks.org/deep-learning/applying-convolutional-neural-network-on-mnist-dataset/

import numpy as np
import tensorflow as tf
import numpy as np
import keras 
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k
import matplotlib.pyplot as plt

# MNIST dataset
print("Mnist dataset")
(X_train,y_train), (X_test,y_test) = mnist.load_data()
print("Success")

# 28 by 28 pixels per image
img_rows, img_cols = 28, 28

# We need to add the 1 so Conva2D will accept images.
if k.image_data_format() == "channels_first":
  X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
  X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  X_train = X_train.reshape(X_train.shape[0],img_rows, img_cols, 1)
  X_test = X_test.reshape(X_test.shape[0],img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols,1)

# We change 28x28 matrix to vector for neural network
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255


# One-hot encoding, so it can be used later
y_train = keras.utils.to_categorical(y_train) #[1,3,5,6,4,3] -> [[0,1,0,0,0,0,0,0,0],
y_test = keras.utils.to_categorical(y_test)

# Input layer
input_shape = Input(shape=input_shape)

# Conv2D layer which convolves image. It basically gives vision to a computer.
# For example, originally picture is stored as numbers 0-255. If computer sees pixel with number 160, it will not tell it anything.
# But convolving runs the picture through matrix (kernel ("filter")) and creates another matrix with kernel + original picture so computer can understand where is edge on the photo, for example
# Animation: https://jeanvitor.com/wp-content/uploads/2017/07/2D_Convolution_Animation.gif
# Matrix size here is (3*3)
layer1 = Conv2D(32, kernel_size=(3,3), activation='relu')(input_shape)

# Same here but one more time with more filters
layer2 = Conv2D(64, kernel_size=(3,3), activation='relu')(layer1)

# This layer picks the max value from a matrix of size (3*3).
layer3 = MaxPooling2D(pool_size=(3,3))(layer2)

# 0.5 Dropout
layer4 = Dropout(0.5)(layer3)

# Flattens the output from layer4.
layer5 = Flatten()(layer4)
# Layer with 250 neurons
layer6 = Dense(250, activation='sigmoid')(layer5)

# Output layer has 10 neurons (because we classify 10 numbers: from 0 to 9)
layer7 = Dense(10, activation='softmax')(layer6)

# Declaring the model
model = tf.keras.Model([input_shape], layer7)

# Compiling
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# Fitting
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Predicting probabilities
# This predicts probabilities of what model thinks is seen on photo.
# So, if picture has 2, the probability list for this picture will look like (theoretical example):
# [0.03, 0.84, 0.026, 0.023, 0.054, 0.01, 0.085, 0.043, 0.027, 0.025]
# So, it predicts with probability of 0.03 that picture shows "1", 0.84 that it is "2", 0.026 that it is "3", ...
predicted_probabilities = model.predict(X_test)

# Some debugs, not neccesary, you can uncomment if you want,
# print("Shape: ", predicted_probabilities.shape)
# print("PRedicted Probab: ", predicted_probabilities[:10])
# print("Y_test: ", y_test[:10])

# Predicted (y) output 
y_pred = np.argmax(predicted_probabilities, axis=1)

# Real (original) y. (Real answer what model should theoretically predict)
y_true = np.argmax(y_test, axis=1)

# If y_pred is same as y_true = it predicted correct.
correct_indices = np.where(y_pred == y_true)[0]

# Otherwise, if y_pred is not same as y_true, it predicted wrong. This is needed in plotting.
incorrect_indices = np.where(y_pred != y_true)[0]

# # Debug prints again
# print("y_pred: ", y_pred)
# print("y_true: ", y_true)
# print("Correct indices: ", correct_indices)
# print("incorrect indices: ", incorrect_indices)

# Amount of correct indices
print(f"Correct: {len(correct_indices)}, Incorrect: {len(incorrect_indices)}")

# Plot correctly predicted
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(f"Predicted {y_pred[correct]}, Actual {y_true[correct]}")  
plt.tight_layout()


# Plot incorrectly predicted
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(f"Predicted {y_pred[incorrect]}, Actual {y_true[incorrect]}")
    
plt.tight_layout()
plt.show()