import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

data = pd.read_csv('HW02/Data/HeightWeight20.csv')

X = data[['Height']].values
y = data[['Weight']].values
param_grid = {'fit_intercept': [True, False]}

def one_cross_Validation(X, y, k_splits, random_state):
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
    model = LinearRegression()
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-mse_scores)
    return rmse_scores

def Nested_cross_Validation(X, y, k_splits, random_state):
    inner_cv = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
    outer_cv = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
    model = LinearRegression()
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='neg_mean_squared_error')
    nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='neg_mean_squared_error')
    
    return np.sqrt(-nested_scores)

seeds = range(1, 100)

nested_scores = Nested_cross_Validation(X, y,5,1)
cross_score = one_cross_Validation(X,y,5,1)

all_cross_score = []
all_nested_scores = []
for seed in seeds:
    cross_score = one_cross_Validation(X,y,5,seed)
    nested_scores = Nested_cross_Validation(X, y,5,seed)
    all_cross_score.append(cross_score)
    all_nested_scores.append(nested_scores)

mean_CV = np.mean(all_cross_score)
std_CV = np.std(all_cross_score)

mean_nested = np.mean(all_nested_scores)
std_nested = np.std(all_nested_scores)

# แสดงค่า RMSE และ standard deviation
print(f'one cross validation:\n mean = {mean_CV} \n Std = {std_CV}')
print(f'nested cross validation:\n mean = {mean_nested} \n Std = {std_nested}')