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

def Holdout(X,y,TrainSize,seed):
    Xtrain, X_test,Ytrain,Y_test = train_test_split(X,Y,train_size=TrainSize,random_state=seed)
    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test,y_pred))
    return rmse

TrainSizes = [0.9, 0.8, 0.5, 0.2, 0.1]
seeds = range(1,11) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = {}

for TrainSize in TrainSizes:
    rmses = []
    for seed in seeds:
        rmse = Holdout(X, Y, TrainSize, seed)
        rmses.append(rmse)
    result[TrainSize] = rmses #{ค่า TrainSize:[ค่า RMSE ตาม Seed 1,....Seed 10]}


summary = {}
for TrainSize, rmses in result.items():
    mean_rmse = np.mean(rmses)
    SD_rmse = np.std(rmses)
    summary[TrainSize] = [mean_rmse,SD_rmse]

for TrainSize in TrainSizes:
    mean, std = summary[TrainSize] 
    print(f'Train size: {TrainSize*100:.0f}% - Mean RMSE: {mean:.4f}, Std RMSE: {std:.4f}')



