import matplotlib.pyplot as plt
import numpy as np

# X และ y ตัวอย่าง
X = [0, 2]
y = [0, 2]

m = len(y)
print("จำนวนข้อมูล =", m)

iterations = 100 # จำนวนรอบการทำซ้ำ

# ฟังก์ชันคำนวณ cost
def cost_function(X, y, theta0, theta1):
    m = len(y)
    cost = 0
    for i in range(m):
        y_pred = theta0 + theta1 * X[i]
        cost += (y_pred - y[i]) ** 2
    totalcost = (1 / (2 * m)) * cost
    return totalcost

# ฟังก์ชัน gradient descent
def gradient_descent(X, y, theta0, theta1, alpha, iterations):
    m = len(y)
    cost_history = []
    theta0_history = []
    theta1_history = []

    for _ in range(iterations):
        sum_errors_theta0 = 0
        sum_errors_theta1 = 0

        for i in range(m):
            y_pred = theta0 + theta1 * X[i]
            error = y_pred - y[i]

            # หาค่าผลรวม Gradient
            sum_errors_theta0 += error
            sum_errors_theta1 += error * X[i]

        # ปรับค่า theta0 และ theta1
        theta0 -= alpha * (1 / m) * sum_errors_theta0
        theta1 -= alpha * (1 / m) * sum_errors_theta1 

        # เก็บค่าเพื่อเอาไป plot กราฟ
        cost_history.append(cost_function(X, y, theta0, theta1))
        theta0_history.append(theta0)
        theta1_history.append(theta1)

    return theta0, theta1, cost_history, theta0_history, theta1_history

# รัน gradient descent ด้วยค่า learning rate ต่าง ๆ
learning_rates = [0.01, 0.1, 0.5, 1.0]
colors = ['r', 'g', 'b', 'orange']

plt.figure(figsize=(13, 10))

# สร้างช่วงข้อมูลสำหรับ contour plot
randomW1 = np.linspace(-7, 9, 100)
box = []
for i in randomW1:
    box.append(cost_function(X, y, 0, i))

# แสดงกราฟ contour
plt.subplot(2, 2, 3)
plt.plot(randomW1, box, label="MSE")
plt.xlabel("w1")
plt.ylabel("MSE")
plt.title("Contour of Cost Function")

# แสดงเส้นทางของ gradient descent ด้วยค่า learning rate ต่าง ๆ บนกราฟ contour
for alpha, color in zip(learning_rates, colors):
    theta0, theta1, cost_history, theta0_history, theta1_history = gradient_descent(X, y, 0, 0, alpha, iterations)
    plt.plot(theta1_history, cost_history, color=color, label=f'Learning Rate = {alpha}')
    plt.scatter(theta1_history, cost_history, color=color, s=10)

plt.legend()

# แสดงผลลัพธ์การเรียนรู้ด้วย gradient descent
theta0, theta1, cost_history, theta0_history, theta1_history = gradient_descent(X, y, 0, 0.5, alpha, iterations)

Y_pred = [theta0 + theta1 * x for x in X]

plt.subplot(2, 2, 1)
plt.scatter(X, y, color="blue", label="Data") 
plt.plot(X, Y_pred, color="green", label="Linear fit") 
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression (Gradient Descent)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction")

plt.tight_layout()
plt.show()
