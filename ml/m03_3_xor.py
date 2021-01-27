from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
#1
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2
model = LinearSVC()

#3
model.fit(x_data,y_data)

#4

y_pred = model.predict(x_data)
print(x_data,"'s result : ",y_pred)


result = model.score(x_data,y_data)
print('modle_socore : ',result)

acc = accuracy_score(y_data,y_pred)
print('accuracy_score : ',acc)