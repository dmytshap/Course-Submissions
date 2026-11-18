import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

#a) 
# Download and upload the data set breast-cancer-wisconsin.data.txt enclosed with this exercise
data = pd.read_csv('wisconsin_data.txt')


print("Before dropping non-numeric values", data.shape[0])
#b 
# Drop the missing or non-numeric values
# By observing dataset, we can see that some values are "?".
# I change "?" to NaN. Then I drop the NaN. 
data = data.replace("?", np.nan)

data = data.dropna(how='any',axis=0)  # Dropping rows that inclued NaN.

print("After: ", data.shape[0])

#c
# Drop the ID column
print("Before ID column drop: (shape) ", data.shape)

print("Before ID column drop: (len) ", len(data.columns))

data = data.drop('id',axis=1)

print("After ID column drop: (shape) ", data.shape)

print("After ID column drop: (len) ", len(data.columns))

#d) Create features and labels arrays X and y
#Features
X = data.drop(columns='class') # All columns beside class
#Labels
y = data['class']  # Just choosing the class column

#e)
# Scale/Standardize the features array X (use Minmax or Standard scaling) (scaling chosen)
# I fit and transform the data with one call. Fit is needed for transform to work.
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Scaling done")

#f) 
# Transform the feature array y to a binary array 0 or 1
# We can see, that 'class' column contains only 2 and 4.
# I replace 2 as 0 and 4 as 1.
print("Class uniques values:", data['class'].unique())
y = y.map({2: 0, 4: 1})


#g)
# Split the arrays into training and test arrays (4 resulting arrays)
X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Splitted in 4 arrays.")

# Check that created arrays are actually arrays and not lists.
if isinstance(X_train,np.ndarray):
    print("It is an numpy array. ")
