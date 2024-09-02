# 1. นำเข้าไลบรารีที่จำเป็น
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 2. เตรียมข้อมูล
digits = load_digits()
X = digits.data
y = digits.target

# แบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def standardization(X):
    mean_x = np.mean(X)
    std_x = np.std(X)
    X_sd = (X - mean_x) / std_x
    return X_sd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    m = y.shape[0]
    loss = - (1/m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    return loss

def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    # Initialize weights (n+1 สำหรับ bias)
    weights = np.zeros((n, 1))
    bias = 0
    loss_history = []

    for i in range(iterations):
        # การคำนวณ Z
        Z = np.dot(X, weights) + bias
        # การคำนวณการคาดการณ์
        y_pred = sigmoid(Z)
        # การคำนวณ loss
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)
        # การคำนวณ gradients
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        # การปรับค่าพารามิเตอร์
        weights -= learning_rate * dw
        bias -= learning_rate * db

        if i % 1000 == 0:
            print(f"Iteration {i}: Loss = {loss}")

    return weights, bias, loss_history

X_train = standardization(X_train)
X_test = standardization(X_test)

# เลือกเฉพาะตัวเลข 0 และ 1
binary_filter = (y_train == 0) | (y_train == 1)
X_train_binary, y_train_binary = X_train[binary_filter], y_train[binary_filter].reshape(-1, 1)

binary_filter_test = (y_test == 0) | (y_test == 1)
X_test_binary, y_test_binary = X_test[binary_filter_test], y_test[binary_filter_test].reshape(-1, 1)

# ฝึกโมเดล Logistic Regression สำหรับ Binary Classification
weights_binary, bias_binary, loss_history_binary = gradient_descent(X_train_binary, y_train_binary, learning_rate=0.1, iterations=10000)

# ทำนายผลการทดสอบ
Z_test_binary = np.dot(X_test_binary, weights_binary) + bias_binary
y_pred_binary = sigmoid(Z_test_binary)
y_pred_class_binary = (y_pred_binary >= 0.5).astype(int)

# คำนวณความแม่นยำ
accuracy_binary = np.mean(y_pred_class_binary == y_test_binary)
print(f"Binary Classification Accuracy: {accuracy_binary * 100:.2f}%")

# ตัวอย่างการพล็อตภาพจากข้อมูล digits
num_images = 10
fig, axes = plt.subplots(1, num_images, figsize=(10, 4))

for i in range(num_images):
    axes[i].imshow(X[i].reshape(8, 8), cmap='gray')
    axes[i].set_title(f"Label: {y[i]}")
    axes[i].axis('off')

plt.show()

# นำ weights จากโมเดล binary มาพล็อตเป็นภาพ
weights_binary_image = weights_binary.reshape(8, 8)

plt.imshow(weights_binary_image, cmap='hot', interpolation='nearest')
plt.title('Weights for Binary Classification (0 vs 1)')
plt.colorbar()
plt.show()




