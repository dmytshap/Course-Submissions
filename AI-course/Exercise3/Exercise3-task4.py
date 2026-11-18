'''
Exercise3-task4.py

4. Begin with a network that has one hidden layer. Test and evaluate the network's
performance with varying nodes: 5, 10, up to 30 nodes
'''

import pandas as pd
import torch as torch
import torch.nn as nn


# Import processed data 
from data_preprocess import X_train, X_test, y_train, y_test

# Import model
from model import NeuralNetwork, model_train, evaluate_model_train, evaluate_model_test



nodes = [1,5,10,15,20,25,30]

result = []


for node in nodes:
    
    # This works following way:
    # Here nodes is [1,5,10,15]. On first iteration, it will choose 1.
    # This 1 is then placed in list ([node]) which is treated as one layer with one node.
    # Next it will choose 5, which will become [5], which will mean 1 layer with 5 nodes. 

    model = NeuralNetwork(
                    hidden_layers=[node],
                    activation_function=nn.ReLU(),
                    dropout_rate=0.0) #no need for dropout in this task.
    
    model_train(
        model,
        X_train,
        y_train,
        num_epochs=1000, # This can be adjusted as wanted / needed
        learning_rate=0.001 # This can be adjusted as wanted / needed
    )
    
    train_accuracy = evaluate_model_train(
        model,
        X_train,
        y_train
    )
    
    test_accuracy = evaluate_model_test(
        model,
        X_test,
        y_test
    )
    
    result.append({
        "Hidden neurons": node,
        "Train accuracy": train_accuracy * 100,
        "Test accuracy": test_accuracy * 100
    })
    
    
    results_df = pd.DataFrame(result)
    
print("Task 4 results:")
print(results_df)
