
import numpy as np

D = np.array([range(1,11),range(11,21),range(21,31),range(31,41)])
D = D.T

print(D)

def split_x(D,size,y_cols):
    x , y = [] , []
    for i in range(len(D)):
        x_end_number = i + size
        y_end_number = x_end_number + y_cols 
        if y_end_number > len(D) :
            break
        tem_x = D[i : x_end_number,:]
        tem_y = D[x_end_number:y_end_number, :]  # 뒤 숫자에 따라 y가 변한다 -1 = (1개씩), : = (한 행)
        x.append(tem_x)
        y.append(tem_y)
    return np.array(x),np.array(y)
    
x, y = split_x(D,3,1)

print(x)
print('----------------')
print(y)
print(y.shape)


# 한면에서 다음 한행(:) 다음 한열(-1) 을 확인



