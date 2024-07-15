import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV


# โหลดข้อมูล sin_noiseless_10sample.csv
data = pd.read_csv('/Users/jittapat.a/Documents/code/ปี4/ML/HW03/archive/sin_noiseless_80sample.csv')
  
X = data.iloc[:, 0].values.reshape(-1, 1)# ฟีเจอร์ทั้งหมดที่ไม่ใช่ y
y = data['y'].values  # คอลัมน์ y

# print(X)
# print(y)
# กำหนดพารามิเตอร์ที่ต้องการค้นหา
param_grid = {'polynomialfeatures__degree': [1, 2, 3, 4, 5, 6, 7, 8]}

def one_cross_Validation(X, y, k_splits, random_state):
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
    model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-mse_scores)
    return rmse_scores

def Nested_cross_Validation(X, y, k_splits, random_state):
    inner_cv = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
    outer_cv = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
    model = make_pipeline(PolynomialFeatures(), LinearRegression())
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='neg_mean_squared_error') #ปรับ hyperparameter ของ model ด้วย GridSearchCV
    nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='neg_mean_squared_error')

    best_params = grid_search.fit(X, y).best_params_
    return np.sqrt(-nested_scores), best_params

seeds = range(1, 11)

all_cross_score = []
all_nested_scores = []
best_params_list = []

for seed in seeds:
    cross_score = one_cross_Validation(X, y, 5, seed)
    nested_scores, best_params = Nested_cross_Validation(X, y, 5, seed)
    all_cross_score.append(cross_score)
    all_nested_scores.append(nested_scores)
    best_params_list.append(best_params)

mean_CV = np.mean(all_cross_score)
std_CV = np.std(all_cross_score)

mean_nested = np.mean(all_nested_scores)
std_nested = np.std(all_nested_scores)


# แสดงค่า RMSE และ standard deviation
print(f'one cross validation:\n mean = {mean_CV} \n Std = {std_CV}')
print(f'nested cross validation:\n mean = {mean_nested} \n Std = {std_nested}')

# แสดงค่า hyperparameter ที่ดีที่สุดที่พบ
print(f'Best hyperparameters found in Nested Cross Validation for each seed:')
for i, params in enumerate(best_params_list, start=1):
    print(f'Seed {i}: {params}')

plt.title("sin_noiseless_80sample")
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
