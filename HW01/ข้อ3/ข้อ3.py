import matplotlib.pyplot as plt
import numpy as np

# X = [1,2,3,4,5]
# y = [1,3,2,5,4]
X = [0,2]
y = [0,2]

m = len(y)
print("จำนวนข้อมูล = ",m)


alpha = [0.1, 0.5, 0.7, 1.001] # learning rate 0.5
iterations = 35 # จำนวนรอบการทำซ้ำ

#ค่าเริ่มต้น w0 กับ w1
theta0 = 0 #กำหนดให้ theta0 (intercept) เริ่มต้น = 0
theta1 = -1 #กำหนดให้ theta1 (slope) เริ่มต้น = 0



def cost_function(X, y, theta0 ,theta1):
    m = len(y)
    cost = 0
    for i in range(m):
        y_pred = theta0 + theta1 * X[i]
        cost += (y_pred - y[i]) ** 2
    totalcost = (1/2*m)*cost
    return totalcost


def gradient_descent(X, y, theta0, theta1, alpha, iterations):
    m = len(y)
    cost_history = [cost_function(X, y, theta0, theta1)]
    theta0_history = [theta0]
    theta1_history = [theta1]

    for _ in range(iterations):
        sum_errors_theta0 = 0
        sum_errors_theta1 = 0

        for i in range(m):
            y_pred = theta0 + theta1 * X[i]
            error = y_pred - y[i]
            
            #หาค่าผลรวม Gradient
            #sum_errors_theta0 = sum_errors_theta0 + error
            sum_errors_theta1 = sum_errors_theta1 + error * X[i]

        #ปรับค่า w0,w1 จากสมการ w0(ใหม่) = w0(เก่า) - learningRate * 1/จำนวนข้อมูล * ผลรวม Gradient
        #theta0 -= alpha * (1/m) * sum_errors_theta0 #i-=1 หรือ i=i-1
        theta1 -= alpha * (1/m) * sum_errors_theta1 
        # เก็บค่าเพื่อเอาไป plot กราฟ
        cost_history.append(cost_function(X, y, theta0, theta1))
        theta0_history.append(theta0)
        theta1_history.append(theta1)
        

    return theta0, theta1, cost_history, theta1_history

plt.figure(figsize=(15, 5))

for i,lr in enumerate(alpha):
    # สร้างช่วงข้อมูล 
    box = []
    randomW1 = np.linspace(-1, 3, 100)
    for j in randomW1:
        box.append(cost_function(X, y, 0, j))

    theta0_1, theta1_1, cost_history_1, theta1_history_1 = gradient_descent(X, y, theta0, theta1, lr, iterations)
    plt.subplot(1,4,i+1)
    plt.plot(randomW1,box,label="MSE")
    plt.plot(theta1_history_1, cost_history_1,"-o", color="red", label="theta_val")
    plt.xlabel("w1")
    plt.ylabel("MSE")
    plt.title(f'learning rate = {lr}')

plt.show()

