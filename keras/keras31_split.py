import numpy as np

a = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset) #[item for item in subset]
    print(type(aaa))
    return np.array(aaa)


dataset = split_x(a,size)
print('=================')
print(dataset)

# 시계열 일때 쉽게 나눌수 있다.