
''' 
AI Exercise 5, part 2

Write down and explain the formulas for SOM.
Study and experiment different SOM related functions that you can find. 

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

print("Start")
np.random.seed(42)

# Features
X = np.loadtxt("/IrisData.txt")
# labels
iris_labels = pd.read_csv("/IrisLabels.txt")

print("Shapes")
print(X.shape)
print(iris_labels.shape)

y_text = iris_labels.iloc[:, 4].astype(str).str.strip()

label_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

y_series = y_text.map(label_mapping)

y = y_series.to_numpy(dtype=int)

print(y.shape)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(np.round(X_scaled.mean(axis=0),6))
print(np.round(X_scaled.std(axis=0),6))

'''
SOM creating
'''

SOM = MiniSom(x=8, y=8, input_len=4, sigma=1.0, learning_rate=0.5,neighborhood_function="gaussian")

SOM.random_weights_init(X_scaled)

SOM.train_random(X_scaled, num_iteration=5000)
print("Training finished")

plt.figure(figsize=(8,8))

umatrix = SOM.distance_map()

plt.imshow(umatrix.T,cmap="bone_r")
plt.colorbar(label="Neighbor distance")
plt.title("SOM Matrix")
plt.xlabel("SOM X coordintate")
plt.ylabel("SOM Y coordinate")
plt.show()

plt.figure(figsize=(8,8))

act_resp = SOM.activation_response(X_scaled)

plt.imshow(act_resp.T,cmap="bone_r")
plt.colorbar(label="Amount of samples")
plt.title("SOM Activation Frequency")
plt.xlabel("SOM X coordintate")
plt.ylabel("SOM Y coordinate")
plt.show()