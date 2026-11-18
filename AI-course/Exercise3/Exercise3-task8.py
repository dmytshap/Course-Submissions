'''
Exercise3-task8.py

8. Construct a network with 2 hidden layers, each containing 10 neurons. Try at least 3
different activation functions for the hidden layers. After evaluating, determine which
activation function produced the best results.
'''

import pandas as pd
import torch as torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Import processed data 
from data_preprocess import X_train, X_test, y_train, y_test

# Import model
from model import NeuralNetwork, model_train, evaluate_model_train, evaluate_model_test


result = []

activation_functions = [nn.ReLU(), nn.GELU(), nn.Tanh()]

for one_activation_function in activation_functions:
    
    layers = [10] * 2  
    
    model = NeuralNetwork(
                    hidden_layers=layers,
                    activation_function=one_activation_function,
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
    
    
    
    #predicted values, needed for confusion matrix
    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.round()
    #makes conf matrix for every type of activation function
    conf_matrix = confusion_matrix(y_test, predictions)
     
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=["Benign", "Malignant"]
    )

    disp.plot()
    plt.title(f"Confusion Matrix - {one_activation_function}")
    plt.show()
    
    result.append({
        "Hidden layers": layers,
        "Train accuracy": train_accuracy * 100,
        "Test accuracy": test_accuracy * 100,
        "Activation function": one_activation_function
    })
    
    
    results_df = pd.DataFrame(result)
    
print("Task 8 results:")
print(results_df)
