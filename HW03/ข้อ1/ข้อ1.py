import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('/Users/jittapat.a/Documents/code/ปี4/ML/HW03/archive/sin_noiseless_40sample.csv')

X = data.iloc[:, 0].values.reshape(-1, 1)  # ฟีเจอร์ทั้งหมดที่ไม่ใช่ y และแปลงเป็น array 2 มิติ
y = data['y'].values  # คอลัมน์ y

def linear_Regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return y_pred

y_p = linear_Regression(X, y)

def Polynomial_Regression(X, y, num):
    poly_model = PolynomialFeatures(degree=num)
    X_poly = poly_model.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred_poly = model.predict(X_poly)
    return y_pred_poly

y_p_poly = Polynomial_Regression(X, y, 3)

plt.title("Polynomial Regression")
plt.scatter(X, y, label='Data')
plt.plot(X, y_p, linewidth=3, color="green", label='Linear Regression')
plt.plot(X, y_p_poly, linewidth=3, color="orange", linestyle="--", label='Polynomial Regression (degree 3)')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
