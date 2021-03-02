import numpy as np
import matplotlib.pyplot as plt

def sigmod(x):
    return 1 / (1 + np.exp(-x))
# sigmoid를 사용하면 레이어의 마지막 값에  1 / (1 + np.exp(-x)) 를 곱해주는것이다.
# 나온 값을 binary crossentropy 에서 0.5이상 미만으로 판단 해서 나눠준다
x = np.arange(-5, 5, 0.1)
y = sigmod(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()