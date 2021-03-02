from tensorflow.keras import datasets
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

(x_train, y_train), (x_test,  y_test) = cifar10.load_data()


x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255.

from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

print(vgg16.weights)


vgg16.trainable = False
vgg16.summary()
print(len(vgg16.weights)) # 26
print(len(vgg16.trainable_weights)) # 0

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',patience=15,mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss',factor = 0.1, patience= 5)
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x_train,y_train,epochs=1000,batch_size=16,validation_split=0.2,verbose=1,callbacks=[es,lr])

loss=model.evaluate(x_test,y_test)
print(loss)

# print("가중치의 수 : ", len(model.weights)) # 26 -> 32
# print("동결된 후 훈련되는 가중치의 수 : ", len(model.trainable_weights)) # 0 -> 6

# import pandas as pd

# pd.set_option('max_colwidth',-1)
# layers = [(layer,layer.name, layer.trainable) for layer in model.layers] 
# aaa = pd.DataFrame(layers, columns= ['Layer Type', 'Layer name', 'Layer Trainable'])
# print(aaa)


# [1.2861682176589966, 0.5496000051498413]