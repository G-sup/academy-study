import numpy as np
f = lambda x : x**2 - 4*x + 6

gradient =  lambda x : 2*x - 4

x0 = 10.0
epoch = 1000
learning_rate = 0.001

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

# 최적의 가중치를 구하는 식

for i in  range(epoch) :
    temp = x0 - learning_rate * gradient(x0) 
    x0 = temp
    
    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))







