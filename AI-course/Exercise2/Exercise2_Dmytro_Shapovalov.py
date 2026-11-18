'''

Following Lecture 4, we aim to implement a single-layer neural network, also known as a
Perceptron, to represent the Boolean functions: AND, OR, and NOR (NOT OR).
1. Generate input and output samples (X and y) for each Boolean function (four
samples for each function).
2. Implement a single-layer neural network as described in the lecture slides (with 2
inputs and 1 output). Execute the feedforward operations.
3. Implement backpropagation using the gradient descent algorithm.
4. Train and test the neural network.
5. Display the results.
6. Can a single-layer perceptron represent the XOR Boolean function? Why or why
not?
7. Enhance the neural network to accurately represent the XOR Boolean function.

'''
import numpy as np

#AND:
# A    B    A,B
# 0    0     0
# 0    1     0
# 1    0     0
# 1    1     1


#OR:
# A    B    A,B
# 0    0     0
# 0    1     1
# 1    0     1
# 1    1     1


#NOR:
# A    B    A,B
# 0    0     1
# 0    1     0
# 1    0     0
# 1    1     0

# Inputs
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Outputs
y_and = np.array([0,0,0,1])
y_or = np.array([0,1,1,1])
y_nor = np.array([1,0,0,0])
class Perceptron:
    def __init__(self, learning_rate = 0.01, epochs = 5000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    # Activation function
    def sigmoid(self, x): 
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_derivative(self, x):
        return x * (1-x)
        
        
    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.epochs):            
            for xi, target in zip(X,y):
                
                #Go forward (forward propagation)
                linear_output = np.dot(xi, self.weights) + self.bias #(w*x+b)
                
                # Predicted after forward propagation
                y_pred = self.sigmoid(linear_output)
                
                # Error computed from predicted and original values
                error = y_pred - target  #error
                
                # Backpropagation
                delta = error*self.sigmoid_derivative(y_pred)
                 
                # Update gradient
                self.weights -= self.lr * delta * xi
                self.bias -= self.lr * delta

    def predict(self,X):
        linear_output = np.dot(X,self.weights) + self.bias # For example [0,1] * [0.5, 0.3] +bias (example 0.4) -> 0*0.5 + 1*0.3 + 0.4 -> 0.7
        y_predicted = self.sigmoid(linear_output) # Then 0.3 scales via sigmoid to be a value from 0 to 1 -> ~ 0.66
        return np.where(y_predicted >= 0.5,1,0) # -> 1


and_model = Perceptron(learning_rate=0.01, epochs=5000)
and_model.train(X, y_and)

or_model =  Perceptron(learning_rate=0.01, epochs=5000)
or_model.train(X,y_or)

nor_model =  Perceptron(learning_rate=0.01, epochs=5000)
nor_model.train(X,y_nor)



print("AND PREDICT: ", and_model.predict(X), ", Weigths: ", and_model.weights, ", Bias: ", and_model.bias)

print("OR PREDICT: ", or_model.predict(X), ", Weigths: ", or_model.weights, ", Bias: ", or_model.bias)

print("NOR PREDICT: ", nor_model.predict(X), ", Weigths: ", nor_model.weights, ", Bias: ", nor_model.bias)
