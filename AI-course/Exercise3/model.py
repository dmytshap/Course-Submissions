'''
model.py

This file contains the "constructor" for model that is used to solve tasks 4,5, 7,8

'''

import torch as torch
import torch.nn as nn
import torch.optim as optim
import tqdm

# https://github.com/shakeel-data/breast-cancer-prediction-neural-network-pytorch
# https://codefinity.com/courses/v2/1dd2b0f6-6ec0-40e6-a570-ed0ac2209666/319d3c62-620f-49b1-a976-87b7eb67ce66/70c6c600-8cdf-4371-8fc7-fb497f7430ab
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers, activation_function, dropout_rate):
        super().__init__()
        
        layers = []
        input_features = 9
        
        for neurons in hidden_layers:
            layers.append(nn.Linear(input_features, neurons))
            
            # Needed in every task but especially in task 8. Default = ReLU()
            layers.append(activation_function)
            
            
            # Needed for task 7.
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            input_features = neurons
        
        # Output layer
        layers.append(nn.Linear(input_features, 1))
        
        # Append sigmoid because we use BCELoss()
        layers.append(nn.Sigmoid())

        # Combines (unpacks "*") all info about neural network for PyTorch to use.
        # This is same as Sequential(nn.linear(*input_features*, *neurons*), nn.ReLU(), nn.Linear(*hidden_layers*, *output*), nn.Sigmoid)) for example.
        self.neural_network = nn.Sequential(*layers)
        
    # Forwardpropagation
    def forward(self,x):
        return self.neural_network(x)

'''
Model training
'''
def model_train(model, X_train, y_train, num_epochs, learning_rate):
    
    # Loss function & optimizer
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)

    # tqdm shows progress of for-loop bar
    for epoch in tqdm.tqdm(range(num_epochs), desc="train step", unit="epoch"):
        # We set model in training mode
        model.train()

        # This clears gradients for new iteration.
        # PyTorch by default accumulates gradients:
        # https://www.geeksforgeeks.org/deep-learning/why-do-we-need-to-call-zerograd-in-pytorch/#:~:text=This%20means%20that%20every%20time%20loss%2Ebackward%28%29%20is%20called%2C%20the%20gradients%20are%20added%20to%20the%20grad%20attribute%20of%20the%20parameters%2C%20rather%20than%20replacing%20them%2E
        optimizer.zero_grad()
        
        # Forwardpropagation / calculating outputs
        outputs = model(X_train)
        
        #Calculate loss
        loss = loss_function(outputs, y_train.view(-1,1))

        #Backpropagation
        loss.backward()
        
        # weights update
        optimizer.step()            
            
            
            

'''
Evaluate model on training data
'''

def evaluate_model_train(model,X_train, y_train):
    
    model.eval()
    
    # This disables local gradient to save memory and make computations faster
    # I don't really think we need to use it here, because dataset is small
    # But this was kind of a common thing to use in tutorials I found
    with torch.no_grad():
        
        # Forwardpropagation
        outputs = model(X_train)
        
        # We convert outcomes to 0 or 1
        predicted = outputs.round()
        
        # How many outputs are correct
        correct = (predicted == y_train.view(-1,1)).float().sum()

        # Calculate accuracy
        accuracy = correct/y_train.size(0)
                
    print(f"Accuracy on training data: {accuracy.item() * 100:.2f}%")
    
    # Return accuracy. .item() changes tensor data to normal number
    return accuracy.item()



'''
Evaluate model on test data
'''
## Needed for confusion matrices in task 7 and 8
def evaluate_model_test(model, X_test, y_test):
    
    model.eval()

    with torch.no_grad():
        #Forwardpropagation
        outputs = model(X_test)
        
        # We convert outcomes to 0 or 1
        predicted = outputs.round()
        
        # How many are correct
        correct = (predicted == y_test.view(-1,1)).float().sum()
        
        # Calculate accuracy
        accuracy = correct/y_test.size(0)
        
    print(f"Accuracy on test data: {accuracy.item() * 100:.2f}%")
    
    # Return accuracy. .item() changes tensor data to normal number
    return accuracy.item()