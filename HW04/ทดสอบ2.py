import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("/Users/jittapat.a/Documents/code/ปี4/ML/HW04/HeightWeight100.csv")

# Split data into training and testing sets
train, test = train_test_split(data, test_size=0.2, random_state=1)
predictors = ["Height"]
target = "Weight"

# Define functions for fitting and predicting
def ridge_fit(train, predictors, target, A_Lambda):
    X_train = train[predictors].copy()
    y_train = train[[target]].copy()
    
    x_mean = X_train.mean()
    x_std = X_train.std()
    
    X_train = (X_train - x_mean) / x_std
    X_train["intercept"] = 1
    X_train = X_train[["intercept"] + predictors]

    penalty = A_Lambda * np.identity(X_train.shape[1])
    penalty[0][0] = 0

    B = np.linalg.inv(X_train.T @ X_train + penalty) @ X_train.T @ y_train
    B.index = ["intercept","Height"]
    return B, x_mean, x_std

def ridge_predict(test, predictors, x_mean, x_std, B):
    test_X = test[predictors]
    test_X = (test_X - x_mean) / x_std
    test_X["intercept"] = 1
    test_X = test_X[["intercept"] + predictors]

    y_test_pred = test_X @ B
    return y_test_pred

# List of Lambda values
A_Lambda = [0.01, 0.1, 1, 10, 100, 1000]

# Plotting
plt.figure(figsize=(10, 6))

# Original data points
plt.scatter(train[predictors], train[target], color='blue', label='Train data', alpha=0.5)
plt.scatter(test[predictors], test[target], color='red', label='Test data', alpha=0.5)

# Predict and plot for each Lambda
x_range = np.linspace(data[predictors].min().values[0], data[predictors].max().values[0], 100)
x_range_df = pd.DataFrame(x_range, columns=predictors)
x_range_normalized = (x_range_df - data[predictors].mean()) / data[predictors].std()
x_range_normalized['intercept'] = 1
x_range_normalized = x_range_normalized[["intercept"] + predictors]

for alpha in A_Lambda:
    B, x_mean, x_std = ridge_fit(train, predictors, target, alpha)
    y_pred_range = x_range_normalized @ B
    
    plt.plot(x_range, y_pred_range, label=f'Lambda = {alpha}')

plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Ridge Regression Predictions')
plt.legend()
plt.show()
