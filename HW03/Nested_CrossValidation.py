import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score


# โหลดข้อมูลตัวอย่าง
data = pd.read_csv('HW02/Data/HeightWeight20.csv')

X = data[['Height']].values
y = data[['Weight']].values

# กำหนดพารามิเตอร์ที่ต้องการค้นหา
param_grid = {'fit_intercept': [True, False]}

# การทำ Nested Cross-Validation
def nested_cv(X, y, outer_splits=5, inner_splits=5):
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=1)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=1)

    model = LinearRegression()
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='neg_mean_squared_error')

    nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='neg_mean_squared_error')
    
    return np.sqrt(-nested_scores)

# ทำ Nested Cross-Validation
nested_scores = nested_cv(X, y)

# แสดงค่า RMSE และ standard deviation
print("Nested CV RMSE: ", nested_scores.mean(), "±", nested_scores.std())

