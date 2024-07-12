import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold



data = pd.read_csv('../Data/HeightWeight20.csv')

X = data[['Height']].values
Y = data[['Weight']].values

# การทดลองที่ 2 ทดสอบความเที่ยงตรงของวิธี fold cross-Validation

def cross_Validation(X, y, k_splits, random_state):
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
    model = LinearRegression()
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-mse_scores)
    return rmse_scores

folds = [10, 5, 3, 2]
seeds = range(1, 100)

# สร้าง Dictionary สำหรับเก็บผลลัพธ์
results = {}

# วน loop ตามจำนวน fold 10, 5, 3, 2
for k_splits in folds:
    all_rmse_scores = []
    # วน loop ตามจำนวน seed 1-10
    for seed in seeds:
        rmse_scores = cross_Validation(X, Y, k_splits, seed)# คำนวณค่า RMSE 
        all_rmse_scores.append(rmse_scores)# เก็บค่า RMSE ที่คำนวณได้ลงใน all_rmse_scores
    results[k_splits] = all_rmse_scores# นำค่าที่เก็บได้มาเก็บใน Dictionary results

# คำนวณค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานสำหรับแต่ละจำนวน fold
summary = {}
for k_splits, rmse_scores in results.items():
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    summary[k_splits] = [mean_rmse, std_rmse]


for k_splits in folds:
    mean_rmse ,std_rmse = summary[k_splits]
    print(f'Number of folds: {k_splits} - Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}')