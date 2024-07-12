import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error


data = pd.read_csv('../Data/HeightWeight.csv')


X = data[['Height']].values  
y = data[['Weight']].values    

def resubstitution(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return rmse

def holdout(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return rmse

def cross_Validation(X, y, random_state):
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    model = LinearRegression()
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-mse_scores)
    return rmse_scores

seeds = range(1, 500)
resubstitution_rmse = []
holdout_rmse = []
cv_rmse = []

for seed in seeds:
    # สุ่มข้อมูลออกมา 20 ตัวอย่าง
    np.random.seed(seed)
    indices = np.random.choice(len(X), 20, replace=False)
    X_sampled = X[indices]
    y_sampled = y[indices]
    
    # คำนวณ RMSE สำหรับ Resubstitution
    resubstitution_rmse.append(resubstitution(X_sampled, y_sampled))
    
    # คำนวณ RMSE สำหรับ Holdout
    holdout_rmse.append(holdout(X_sampled, y_sampled, seed))
    
    # คำนวณ RMSE สำหรับ 10-fold Cross Validation
    cv_rmse.append(cross_Validation(X_sampled, y_sampled, seed))

summary = {
    'Resubstitution': [
        np.mean(resubstitution_rmse),
        np.std(resubstitution_rmse)
        ],
    'Holdout': [
        np.mean(holdout_rmse),
        np.std(holdout_rmse)
        ],
    'Cross Validation': [
        np.mean(cv_rmse),
        np.std(cv_rmse)]
}

print("Results Summary")
print("================")
for method, values in summary.items():
    mean_rmse, std_rmse = values
    print(f'{method} - Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}')
