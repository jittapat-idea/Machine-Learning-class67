import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def multivariate_gaussian_pdf(r, mu, sigma):
    D = len(r)
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    term1 = 1 / (np.power(2 * np.pi, D / 2) * np.sqrt(det_sigma))
    term2 = np.exp(-0.5 * (r - mu).T @ inv_sigma @ (r - mu))
    return term1 * term2

# สร้างข้อมูลสำหรับสองคลาส
mu1 = np.array([1, 1])
sigma1 = np.array([[1, 0.5], [0.5, 1]])

mu2 = np.array([-1, -1])
sigma2 = np.array([[1, -0.5], [-0.5, 1]])

# ฟังก์ชันสำหรับสร้างตารางความน่าจะเป็น (likelihood) สำหรับแต่ละคลาส
def calculate_likelihood_grid(mu, sigma, grid_x, grid_y):
    likelihood = np.zeros_like(grid_x)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            r = np.array([grid_x[i, j], grid_y[i, j]])
            likelihood[i, j] = multivariate_gaussian_pdf(r, mu, sigma)
    return likelihood

# สร้างตารางพิกัด x และ y
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# คำนวณ likelihood สำหรับคลาส 1 และคลาส 2
likelihood_class1 = calculate_likelihood_grid(mu1, sigma1, X, Y)
likelihood_class2 = calculate_likelihood_grid(mu2, sigma2, X, Y)

# Posterior คำนวณโดยใช้ Bayes' theorem (ถือว่า priors มีค่าเท่ากัน)
posterior_class1 = likelihood_class1 / (likelihood_class1 + likelihood_class2)
posterior_class2 = likelihood_class2 / (likelihood_class1 + likelihood_class2)

# วาดกราฟ likelihood และ posterior
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot likelihood of class 1
axes[0].contourf(X, Y, likelihood_class1, cmap='Blues')
axes[0].set_title('Likelihood of Class 1')

# Plot likelihood of class 2
axes[1].contourf(X, Y, likelihood_class2, cmap='Reds')
axes[1].set_title('Likelihood of Class 2')

# Plot posterior for class 1
axes[2].contourf(X, Y, posterior_class1, cmap='Purples')
axes[2].set_title('Posterior of Class 1')

# Decision boundary (เมื่อ posterior_class1 = 0.5)
axes[2].contour(X, Y, posterior_class1, levels=[0.5], colors='black')

plt.show()