import numpy as np

D = np.array([range(1,11),range(11,21),range(21,31)])
D = D.T

print(D)

def split_x(D,x_row,y_cols):
    x , y = [] , []
    for i in range(len(D)):
        x_end_number = i + x_row
        y_end_number = x_end_number + y_cols -1
        if y_end_number > len(D) :
            break
        tem_x = D[i : x_end_number,:-1]
        tem_y = D[x_end_number -1 :y_end_number, -1] # 뒤 숫자에 따라 y가 변한다
        x.append(tem_x)
        y.append(tem_y)
    return np.array(x),np.array(y)
    
x, y = split_x(D,3,2)

print(x)
print('----------------')
print(y)

# 한면에서 (y값 제외) 에서 y를 맞춘다