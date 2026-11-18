'''
data_preprocess.py

This file contains Pre-processing
'''

import pandas as pd
import numpy as np
import torch as torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#a) 
# Download and upload the data set breast-cancer-wisconsin.data.txt enclosed with this exercise
data = pd.read_csv('/home/dima/Desktop/University/AI-course/Exercise3/wisconsin_breast_cancer_data.txt')


# I change "?" to NaN. Then I drop the NaN. 
data = data.replace("?", np.nan)

data = data.dropna(how='any',axis=0)  # Dropping rows that inclued NaN.


#c
# Drop the ID column
data = data.drop('id',axis=1)

#d) Create features and labels arrays X and y

#Features
X = data.drop(columns='class') # All columns beside class
#Labels
y = data['class']  # Just choosing the class column


#f) 
# Transform the feature array y to a binary array 0 or 1
y = y.map({2: 0, 4: 1})


#g)
# Split the arrays into training and test arrays (4 resulting arrays)
X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Splitted in 4 arrays.")


scaler = StandardScaler()

# Scale after splitting    
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




# I use PyTorch, so we need to change data to PyTorch tensors.
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train.to_numpy(dtype=np.float32))
y_test = torch.from_numpy(y_test.to_numpy(dtype=np.float32))


print("Preprocess done")