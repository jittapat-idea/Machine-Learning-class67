import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('../Data/HeightWeight100.csv')

X = data[['Height']].values
Y = data[['Weight']].values

# การทดลองที่ 3 ทดสอบความเที่ยงตรง และ ความแม่นยำ เมื่อข้อมูลเพิ่มขึ้น

def Holdout(X,y,TrainSize,seed):
    Xtrain, X_test,Ytrain,Y_test = train_test_split(X,Y,train_size=TrainSize,random_state=seed)
    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test,y_pred))
    return rmse

def cross_Validation(X, y, n_splits, random_state):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = LinearRegression()
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-mse_scores)
    return rmse_scores

folds = [20, 10, 5, 3, 2]
TrainSizes = [0.9, 0.8, 0.5, 0.2, 0.1]
seeds = range(1,11)

# สร้าง Dictionary สำหรับเก็บผลลัพธ์
results_HoldOut = {}
results_CrossV = {}


for TrainSize in TrainSizes:
    rmses = []
    for seed in seeds:
        rmse = Holdout(X, Y, TrainSize, seed)
        rmses.append(rmse)
    results_HoldOut[TrainSize] = rmses

# วน loop ตามจำนวน fold 20, 10, 5, 3, 2
for n_splits in folds:
    all_rmse_scores = []
    # วน loop ตามจำนวน seed 1-10
    for seed in seeds:
        rmse_scores = cross_Validation(X, Y, n_splits, seed)# คำนวณค่า RMSE 
        all_rmse_scores.append(rmse_scores)# เก็บค่า RMSE ที่คำนวณได้ลงใน all_rmse_scores
    results_CrossV[n_splits] = all_rmse_scores# นำค่าที่เก็บได้มาเก็บใน Dictionary results

summary_CV = {}
summary_HO = {}
for n_splits, rmse_scores in results_CrossV.items():
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    summary_CV[n_splits] = [mean_rmse, std_rmse]


for TrainSize, rmses in results_HoldOut.items():
    mean_rmse = np.mean(rmses)
    SD_rmse = np.std(rmses)
    summary_HO[TrainSize] = [mean_rmse,SD_rmse]

print("Results Summary")
print("================")
for k_splits in folds:
    mean_rmse ,std_rmse = summary_CV[k_splits]
    print(f'Number of folds: {k_splits} - Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}')
print("================")
for TrainSize in TrainSizes:
    mean, std = summary_HO[TrainSize] 
    print(f'Train size: {TrainSize*100:.0f}% - Mean RMSE: {mean:.4f}, Std RMSE: {std:.4f}')



