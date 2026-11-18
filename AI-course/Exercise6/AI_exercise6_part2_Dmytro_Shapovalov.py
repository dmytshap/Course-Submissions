'''
Make a convolutional neural network (CNN) classifier for classifying dogs and cats. Use data
from Kaggle (https://www.kaggle.com/c/dogs-vs-cats/data) that is also with this assignment.
Comment your code well explaining each step and report classifying error / accuracy. Submit
a report and program code(s).

'''

'''
This tutorial was used as main source for this Exercise. 
https://www.kaggle.com/code/sachinpatil1280/cats-vs-dogs-image-classification-using-cnn-95/notebook#notebook-container
'''



# imports
# Basic
import os
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
import numpy as np
import pandas as pd

# visuals
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay

# Tensorflow
import tensorflow
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,MaxPooling2D,Dropout,Flatten,BatchNormalization,Conv2D
from keras.callbacks import ReduceLROnPlateau,EarlyStopping

# Pathes to data
train_path = '/Exercise6/cats-dogs/train/'
test_path = '/Exercise6/cats-dogs/test1/'

# Extract name of files
filenames = os.listdir(train_path)

# Extract label from the file name
labels = [x.split(".")[0] for x in filenames]

# Now we can know, that for example file named "dog.9988.jpg" has label "dog".
data = pd.DataFrame({"filename": filenames, "label":labels})

# Inspect what we got
print(data[:-10])

# Input (Learning data)
X = data["filename"]

# What output to expect (Label(s) from data)
labels = data['label']

# Split to train and split
# stratify = tells how data should be splitted. If our data for example has 25% 1s and 75% 0s, it would make sure that all splits have same amount of 1s and 0s.
X_train, X_temp = train_test_split(data, test_size = 0.2, stratify=labels, random_state = 42)

label_test_val = X_temp['label']

# X_test gets 10% of whole dataset and X_validation gets also 10% of all dataset
X_test, X_validation = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state=42)

# Shapes
print('The shape of train data',X_train.shape)
print('The shape of test data',X_test.shape)
print('The shape of validation data',X_validation.shape)

# Constants 
image_size = 128
image_channels = 3
batch_size = 32

# This will be used to make dataset of images
train_data_generator = ImageDataGenerator(rescale=1./255,
                                    rotation_range = 15,
                                    horizontal_flip = True,
                                    zoom_range = 0.2,
                                    shear_range = 0.1,
                                    fill_mode = 'reflect',
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1)

test_data_generator = ImageDataGenerator(rescale=1./255)

print("train_data_generator and test_data_generator done")

# Now we apply this train_data_generator to make datasets based on X_train, X_validation and X_test.
# So, this train_generator will give model 80% of data in batches of 32, batch after batch for training.
train_generator = train_data_generator.flow_from_dataframe(X_train,
                                                    directory = '/Exercise6/cats-dogs/train/',
                                                    x_col= 'filename',
                                                    y_col= 'label',
                                                    batch_size = batch_size,
                                                    target_size = (image_size,image_size)
                                                   )

# Same here, but 10% of data and data is used for validation.
validation_generator = test_data_generator.flow_from_dataframe(X_validation, 
                                                 directory = '/Exercise6/cats-dogs/train/',
                                                 x_col= 'filename',
                                                 y_col= 'label',
                                                 batch_size = batch_size,
                                                 target_size = (image_size,image_size),
                                                 shuffle=False
                                                )
# 10% of data and used for testing the final model.
test_generator = test_data_generator.flow_from_dataframe(X_test, 
                                                  directory = '/Exercise6/cats-dogs/train/',
                                                  x_col= 'filename',
                                                  y_col= 'label',
                                                  batch_size = batch_size,
                                                  target_size = (image_size,image_size),
                                                  shuffle=False
                                                 )


#Model
model = Sequential()

# Input Layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape = (image_size,image_size,image_channels))) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block 1 
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block 3
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Fully Connected layers 
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output layer
model.add(Dense(2,activation='softmax'))

print("summary next")
model.summary()

# When model stoppes improving, call this so the learning rate will be reduced.
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                            patience=2,
                                            factor=0.5,
                                            min_lr = 0.00001,
                                            verbose = 1
                                            )

# Early stopping
early_stoping = EarlyStopping(monitor='val_loss',patience= 5,restore_best_weights=True,verbose=0)

# Compiling the model.
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fit the model
# This tells model to start learning on train_generator dataset and use validation_generator dataset
cat_dog_classify = model.fit(train_generator,
                             validation_data = validation_generator,
                             callbacks=[early_stoping,learning_rate_reduction],
                             epochs=30
                             )
# Saving model
model.save("model_v2.keras")
model.save("model_v2.h5")

# Erros from model
error = pd.DataFrame(cat_dog_classify.history)

# Plots for  Loss and Accuracy

# Plot for Loss
plt.figure(figsize=(18,5),dpi=200)
sns.set_style('darkgrid')

plt.subplot(121)
plt.title('Cross Entropy Loss',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.plot(error['loss'])
plt.plot(error['val_loss'])


# Plot for Accuracy
plt.subplot(122)
plt.title('Classification Accuracy',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.plot(error['accuracy'])
plt.plot(error['val_accuracy'])

plt.show()


# Prediction 
result = model.predict(test_generator,batch_size=batch_size, verbose=1)

# Predicted outputs
y_pred = np.argmax(result,axis=1)

# Real outputs
y_true = test_generator.labels

#Calculate loss and accuracy
loss, acc = model.evaluate(test_generator, batch_size=batch_size, verbose=1)

# Loss and ACC.
print("LOSS: ", loss)
print("ACC: ", acc)

labels =['Cat','Dog']
# Precisin, Recall, F1-score and support.
print(classification_report(y_true, y_pred,target_names=labels))

