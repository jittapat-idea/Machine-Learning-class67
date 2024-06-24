import numpy as np
import matplotlib.pyplot as plt

# สร้างข้อมูลตัวอย่าง
X = np.array([1, 2])
y = np.array([0, 2])

print("X:\n", X)
print("y:\n", y)

# จำนวนข้อมูล
n = np.size(X)

# คำนวณค่าเฉลี่ยของ x และ y
m_x = np.mean(X)
m_y = np.mean(y)

# คำนวณ cross-deviation และ deviation เกี่ยวกับ x
SS_xy = np.sum(y * X) - n * m_y * m_x
SS_xx = np.sum(X * X) - n * m_x * m_x

# คำนวณค่าสัมประสิทธิ์ (coefficients)
w1 = SS_xy / SS_xx #ค่าความชัน
w0 = m_y - w1 * m_x#ค่าจุดตัดแกน y

print("ค่าสัมประสิทธิ์(ค่าความชัน) = ", w1)
print("bias(ค่าจุดตัดแกน y) = ", w0)

# คำนวณค่าที่ทำนาย (predicted values)
Y_pred = w0 + w1 * X
print("y ทำนาย = ", Y_pred)

# คำนวณค่า cost function โดยใช้ Mean Squared Error (MSE) วัดค่าเสียหายระหว่างค่าที่ทำนายได้ y ทำนาย กับ y จริง
mse = np.mean((y - Y_pred) ** 2)
print("MSE =", mse)

# แสดงกราฟของข้อมูลและเส้น Linear Regression
plt.figure(figsize=(14, 6))
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, Y_pred, color="green", label="Linear fit")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
