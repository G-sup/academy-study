import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np 
from numpy.core.fromnumeric import cumsum, shape
from sklearn import datasets 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA # decomposition 분해 
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train,x_test, axis=0)
y = np.append(y_train,y_test, axis=0)

print(x.shape)
x = x.reshape(70000,-1)

pca =PCA(n_components=713)
x = pca.fit_transform(x)
x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)



print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)


#2
model = Sequential()
model.add(Dense(128, input_shape=(713,)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(34))
model.add(Dense(34))
model.add(Dense(10,activation='softmax'))

#3
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2,verbose=1)

#4
loss= model.evaluate(x_test,y_test)
print(loss)


y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=-1)
y_test = np.argmax(y_test,axis=-1)

print(y_pred)
print(y_test)

# [0.2995654046535492, 0.9129999876022339]
# [7 2 1 ... 4 5 6]
# [7 2 1 ... 4 5 6]

# [0.4061594307422638, 0.8845000267028809]
# [4 7 9 ... 8 5 5]
# [4 7 9 ... 8 5 5]