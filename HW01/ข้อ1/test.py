import numpy as np
import matplotlib.pyplot as plt

# สร้างข้อมูลตัวอย่าง
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# เพิ่ม column ของ 1 เข้าไปที่ X เพื่อให้รวม bias term (w0) ได้
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# ฟังก์ชันการคำนวณค่าเสียหาย (Cost Function)
def compute_cost(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

# ฟังก์ชัน Gradient Descent
def gradient_descent(X, y, w, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    w_history = np.zeros((iterations, w.shape[0]))
    
    for it in range(iterations):
        prediction = np.dot(X, w)
        w = w - (1/m) * learning_rate * (X.T.dot(prediction - y))
        cost_history[it] = compute_cost(X, y, w)
        w_history[it, :] = w.T
        
    return w, cost_history, w_history

# เริ่มต้นค่า weights ด้วยค่า 0
w_initial = np.zeros((X_b.shape[1], 1))

# กำหนด learning rate และจำนวน iterations
learning_rate = 0.1
iterations = 50

# ใช้ Gradient Descent
w_gd, cost_history, w_history = gradient_descent(X_b, y, w_initial, learning_rate, iterations)

# การวาดกราฟ
plt.figure(figsize=(14, 6))

# กราฟที่ 1: การทำนายของ Linear Regression
plt.subplot(1, 2, 1)
plt.plot(X, y, "b.")
plt.plot(X, X_b.dot(w_gd), "r-", linewidth=2, label="Current hypothesis")
plt.xlabel("Size (feet^2)")
plt.ylabel("Price $ (in 1000s)")
plt.title("Training data with linear regression fit")
plt.legend()

# กราฟที่ 2: การลดลงของค่า Cost ในระหว่างการฝึก Gradient Descent
plt.subplot(1, 2, 2)
theta_0_vals = np.linspace(-10, 10, 100)
theta_1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta_0_vals), len(theta_1_vals)))

for i in range(len(theta_0_vals)):
    for j in range(len(theta_1_vals)):
        t = np.array([theta_0_vals[i], theta_1_vals[j]])
        t = t.reshape(-1, 1)
        J_vals[i, j] = compute_cost(X_b, y, t)

theta_0_vals, theta_1_vals = np.meshgrid(theta_0_vals, theta_1_vals)
plt.contour(theta_0_vals, theta_1_vals, J_vals.T, levels=np.logspace(-2, 3, 20), cmap='viridis')
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(w_history[:, 0], w_history[:, 1], 'r-x')
plt.title("Cost function contour with gradient descent path")

plt.show()
