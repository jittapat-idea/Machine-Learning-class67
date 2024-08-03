import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("/Users/jittapat.a/Documents/code/ปี4/ML/HW04/HeightWeight100.csv")

X = data[['Height']].values
y = data[['Weight']].values

# Split data into training and testing sets
train, test = train_test_split(data, test_size=0.2, random_state=1)
predictors = ["Height"]
target = "Weight"

# Prepare training data
X_train = train[predictors].copy()
y_train = train[[target]].copy()
x_mean = X_train.mean()
x_std = X_train.std()

print("X mean = ", x_mean)
print("X std = ", x_std)
X_train = (X_train - x_mean) / x_std
X_train["intercept"] = 1
X_train = X_train[["intercept"] + predictors]

# Ridge Regression parameters
A_Lambda = 2
I = np.identity(X_train.shape[1])
penalty = A_Lambda * I

# Calculate coefficients
B = np.linalg.inv(X_train.T @ X_train + penalty) @ X_train.T @ y_train
B.index = ["intercept", "Height"]

# Prepare testing data
X_test = test[predictors]
X_test = (X_test - x_mean) / x_std
X_test["intercept"] = 1
X_test = X_test[["intercept"] + predictors]

# Predictions
y_train_pred = X_train @ B
y_test_pred = X_test @ B

# Calculate RMSE
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(test[target], y_test_pred))

print(f"Train RMSE: {rmse_train}")
print(f"Test RMSE: {rmse_test}")

# Plotting
plt.figure(figsize=(14, 6))

# Plot training data and predictions
plt.subplot(1, 2, 1)
plt.scatter(train[predictors], y_train, color='blue', label='Training data')
plt.plot(train[predictors], y_train_pred, color='red', label='Model prediction')
plt.title('Train Data and Prediction')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()

# Plot testing data and predictions
plt.subplot(1, 2, 2)
plt.scatter(train[predictors], y_train, color='blue', label='Training data')
plt.scatter(test[predictors], test[target], color='green', label='Testing data')
plt.plot(train[predictors], y_train_pred, color='red', label='Model prediction')
plt.title('Train and Test Data with Prediction')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()

plt.tight_layout()
plt.show()
