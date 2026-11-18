import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

#a)
# Load the data in GD_Example.txt (it includes 500 pairs of data (xi,yi))
data = pd.read_csv(
	'gd_data.txt',
	sep=r"\s+",
)

# Set xpoints and ypoints
xpoints = data.iloc[:, 0]
ypoints = data.iloc[:, 1]

plt.scatter(xpoints, ypoints, label="Data points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("GD Data")
plt.legend()
plt.show()



# c)
# Implement the cost function (loss function)
def cost_function(m,b,xpoints,ypoints):
    N = len(xpoints)
    total = 0
    
    for i in range(N): # Sigma symbol from formula
        mxib = m * xpoints[i]+b # (mx_i+b) part of formula
        yi = ypoints[i] - mxib #y_i-(mx_i+b) part of formula
        
        total += yi**2 #(y_i-(mx_i+b))^2 part of formula
    return total / N  # divide by length of dataset


learning_rate = 0.01 # formula
m = 0.0
b = 0.0

# d)
# Using gradient descent algorithm with 500 iterations, find the best fitting line characterized by: 𝑚𝑥 + 𝑏. (determine m and b)

# dm part of the formula
def dm(m,b,X,y):
    N = len(X)
    
    total = 0
    
    for i in range(N):
        total += -2*X[i]*(y[i]-(m*X[i] +b))
    
    return total / N

#db part of the formula
def db(m,b,X,y):
    N = len(X)
    
    total = 0
    
    for i in range(N):
        total += -2*(y[i]-(m*X[i]+b))
        
    return total / N
    
def gradient_descent(m,b,X,y):
    m = m - dm(m,b,X,y) * learning_rate
    b = b - db(m,b,X,y) * learning_rate

    return m,b

# d)
#Using gradient descent algorithm with 500 iterations, find the best fitting line characterized by: 𝑚𝑥 + 𝑏. (determine m and b)
for i in range(500):    
    m, b = gradient_descent(m,b, xpoints, ypoints)

print("Gradient descent m (slope): " + str(m))
print("Gradient descent b (intercept): " + str(b))
print("Cost: " + str(cost_function(m, b, xpoints, ypoints)))

#b) Plot the data points.
#e) Plot the final fitting line alongside with the scattered data points.
xd = np.linspace(xpoints.min(), xpoints.max(), len(xpoints))   
# here is the mx + b part. 
yd = m*xd+b  
plt.scatter(xpoints, ypoints, label="Data points")
plt.plot(xd, yd, color="red", label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("GD Data")
plt.legend()
plt.show()