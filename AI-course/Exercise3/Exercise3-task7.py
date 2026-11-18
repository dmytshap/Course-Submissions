'''
Exercise3-task9.py

7. Design a network with 4 hidden layers, each having 10 neurons. Introduce dropout at
each layer. Experiment with different dropout rates ranging from 0% to 40%. Based
on your results, identify which dropout rate yielded the best performance.
'''



import pandas as pd

import torch as torch
import torch.nn as nn


# Import processed data 
from data_preprocess import X_train, X_test, y_train, y_test

# Import model
from model import NeuralNetwork, model_train, evaluate_model_train, evaluate_model_test



result = []

amount_of_layers = 4

dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4]

for dropout_rate in dropout_rates: 
    
    layers_with_neurons = [10] * amount_of_layers  
  

    model = NeuralNetwork(
                    hidden_layers=layers_with_neurons,
                    activation_function=nn.ReLU(),
                    dropout_rate=dropout_rate) 
    
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
        "Hidden layers": amount_of_layers,
        "Train accuracy": train_accuracy * 100,
        "Test accuracy": test_accuracy * 100,
        "Dropout rate": dropout_rate
    })
    
    
    results_df = pd.DataFrame(result)
    

print("Task 7 results:")
print(results_df)
