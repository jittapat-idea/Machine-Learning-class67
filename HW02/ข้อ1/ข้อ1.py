import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('../Data/HeightWeight20.csv')

X = data[['Height']].values
Y = data[['Weight']].values
# print('x = ',X)
# print('y = ',Y)


# การทดลองที่ 1 ทดสอบความเที่ยงตรงของวิธี Holdout

# def calculate_rmse(X,y,TrainSize,seed):
#     Xtrain, X_test,Ytrain,Y_test = train_test_split(X,Y,train_size=TrainSize,random_state=seed)
#     model = LinearRegression()
#     model.fit(Xtrain, Ytrain)
#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(Y_test,y_pred))
#     return rmse

# TrainSizes = [0.9, 0.8, 0.5, 0.2, 0.1]
# seeds = range(1,11) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# result = {}

# for TrainSize in TrainSizes:
#     rmses = []
#     for seed in seeds:
#         rmse = calculate_rmse(X, Y, TrainSize, seed)
#         rmses.append(rmse)
#     result[TrainSize] = rmses

# sum = {TrainSize: {'mean_rmse':np.mean(rmses), 'SD_rmse':np.std(rmses)} for TrainSize, rmses in result.items()}

# for TrainSize in TrainSizes:
#     mean = sum[TrainSize]['mean_rmse']
#     std = sum[TrainSize]['SD_rmse']
#     print(f'Train size: {TrainSize*100:.0f}% - Mean RMSE: {mean:.4f}, Std RMSE: {std:.4f}')



# การทดลองที่ 2 ทดสอบความเที่ยงตรงของวิธี fold cross-Validation

def calculate_rmse_for_cv(X, y, n_splits, random_state):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = LinearRegression()
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-mse_scores)
    return rmse_scores

folds = [10, 5, 3, 2]
seeds = range(1, 11)
results = {}

for n_splits in folds:
    all_rmse_scores = []
    for seed in seeds:
        rmse_scores = calculate_rmse_for_cv(X, Y, n_splits, seed)
        all_rmse_scores.extend(rmse_scores)
    results[n_splits] = all_rmse_scores

# คำนวณค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานสำหรับแต่ละจำนวน fold
summary = {n_splits: {'mean_rmse': np.mean(rmse_scores), 'std_rmse': np.std(rmse_scores)} for n_splits, rmse_scores in results.items()}


for n_splits in folds:
    mean_rmse = summary[n_splits]['mean_rmse']
    std_rmse = summary[n_splits]['std_rmse']
    print(f'Number of folds: {n_splits} - Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}')
