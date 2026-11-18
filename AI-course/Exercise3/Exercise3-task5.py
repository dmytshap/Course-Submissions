'''
Exercise3-task5.py

5. Progressively increase the number of layers from 1 up to 10. For each configuration,
evaluate the network's performance. Note: You have the freedom to choose the
number of neurons per layer
'''


import pandas as pd
import torch as torch
import torch.nn as nn

# Import processed data 
from data_preprocess import X_train, X_test, y_train, y_test

# Import model
from model import NeuralNetwork, model_train, evaluate_model_train, evaluate_model_test



result = []


for amount_of_layers in range(1,11): # Starting at 1 layers, ending on 10.
    
    # This works following way:
    # Here, on first itteration amount_of_layers is 1.  It then multiplies [10] by 1. We will get [10] which will be passed as 1 layer with 10 neurons.
    # Next amount_of_layers is 2. We multiply [10] by 2 and we get [10,10] which means 2 layers each containing 10 neurons. 

    layers = [10] * amount_of_layers
    
    model = NeuralNetwork(
                    hidden_layers=layers,
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
        "Hidden layers": amount_of_layers,
        "Train accuracy": train_accuracy * 100,
        "Test accuracy": test_accuracy * 100
    })
    
    
    results_df = pd.DataFrame(result)
    
print("Task 5 results:")
print(results_df)