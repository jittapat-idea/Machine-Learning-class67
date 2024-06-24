import matplotlib.pyplot as plt
import numpy as np

X = np.array([[2104,1416,1534,852],
             [5,   3,   3,   2  ],
             [1,   2,   2,   1  ],
             [45,  40,  30,  36 ]])

y = np.array([460, 232, 315, 178])
print("x = ",X.shape)
print("y = ",y.shape)

# เพิ่ม column ของ 1 เข้าไปที่ X เพื่อให้รวม bias term (w0) ได้
X_b = np.c_[np.ones((X.shape[0], 1)), X]
