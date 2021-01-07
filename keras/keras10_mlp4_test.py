# x는 (100,5) 데이터 임의로 구성
# y는 (100,2) 데이터 임의로 구성
# 모델을 완성

# 다만든 친구는 predict 일부값 출력

import numpy as np
#1 데이터
x = np.array([range(100),range(301,401),range(1,101),range(501,601),range(801,901)])
y = np.array([range(711, 811), range(1,101)])
print(x.shape) # (5, 100)
print(y.shape) # (2, 100)

x_pred2 = np.array([100, 401, 101, 601, 901]) #일부값을 출력하기 위해선 새로 데이터를 넣어준다.
# x_pred2 = np.transpose(x_pred2) # transpose를 사용못한다
print('x_pres2.shape : ', x_pred2.shape)
x_pred2 = x_pred2.reshape(1, 5) # transpose는 행열을 변화주는 것이기 때문에 열이없는 스칼라 상태에선 reshape를 사용한다
print('x_pres2.shape : ', x_pred2.shape) 

x = np.transpose(x)
y = np.transpose(y)
print(x.shape) #(100, 5)
print(y.shape) #(100, 2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

print(x_train.shape) # (80,5)
print(y_train.shape) # (80,2)

#2 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=5)) 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2)) 


#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2)

#4 평가 예측
loss,mae = model.evaluate(x_test, y_test)
print('loss : ',loss)
print('mae : ', mae)

y_predict = model.predict(x_test) 
print(y_predict)



from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

y_pred2 = model.predict(x_pred2) #새로운 predict를 사용해도 된다
print(y_pred2)
#[811.00244 101.0032 ]


# loss :  2.001794996431272e-08
# mae :  0.00010656565427780151
# [[719.0001       8.999896  ]
#  [803.9998      93.9999    ]
#  [715.           5.0000024 ]
#  [716.00006      5.999808  ]
#  [763.          52.999924  ]
#  [751.99994     41.99996   ]
#  [711.           0.99995166]
#  [783.9998      74.0001    ]
#  [798.9999      89.00006   ]
#  [778.9996      69.00012   ]
#  [736.00006     25.999884  ]
#  [728.99976     19.000193  ]
#  [736.99994     26.99996   ]
#  [739.99976     30.000072  ]
#  [776.9997      67.000084  ]
#  [761.0001      50.999805  ]
#  [790.9999      81.00004   ]
#  [756.          45.999973  ]
#  [749.          38.999943  ]
#  [768.99976     59.000034  ]]
# RMSE :  0.00014148480251457337
# R2:  0.9999999999746714
