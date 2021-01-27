import numpy as np
from numpy.core.fromnumeric import reshape, size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dropout,Dense,GRU,Input,Conv1D ,Flatten ,MaxPool1D
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras import activations
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras.backend as K

df = pd.read_csv('./dacondata/train/train.csv', index_col=[0,1,2], header=0) 

df = df[['DHI','DNI','RH','T','TARGET']]


# def univariate_data(dataset, start_index,end_index, history_size, target_size):
#     data = []
#     labels - []

#     start_index = start_index + history_size
#     if end_index is None:
#         end_index = len(dataset) - target_size
    
#     for i in range(start_index, end_index):
#         indices = range(i-history_size, i)
#         # Reshape data from (history_size,) to (history_size, 1)
#         data.append(np.reshape(dataset[indices],(history_size,1)))
#         labels.append(dataset[i+target_size])
#     return np.array(data), np.array(labels)

TRAIN_SPLIT = 4000

dataset = df.values

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size,step, single_step = False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i-history_size, i,step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)
past_history = 960
future_target = 96
STEP = 1


x_train, y_train = multivariate_data(dataset, dataset[:,1], 0, TRAIN_SPLIT, past_history, future_target,STEP, single_step = True)
x_val, y_val = multivariate_data(dataset, dataset[:,1], TRAIN_SPLIT, None, past_history, future_target,STEP, single_step = True)

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, train_size = 0.8, random_state=104)


print(x_train) # (34040, 240, 5)
print('=============================')
print(y_train) # (34040,)
print('=============================')
print(x_val) # (34040, 240, 5)
print('=============================')
print(y_val) # (34040,)

# df_test = []

# for i in range(81):
#     file_path = './test/' + str(i) + '.csv'
#     df2 = pd.read_csv(file_path, index_col=[0,1,2], header=0) 
#     df2 = df2[['DHI','DNI','RH','T','TARGET']]
#     df2 = df2.dropna(axis=0).values
#     x_pred = df2[-96:,:]
#     df_test.append(x_pred)
    
# X_test = np.concatenate(df_test)
# x_pred = X_test.reshape(-1, 4, 5)
# print(x_pred)


'''


from tensorflow.keras.backend import mean, maximum

qunatile_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def quantile_loss(q, y, pred):
    err=(y-pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

for q in qunatile_list:
    model=Sequential()
    model.add(GRU(128,activation='relu',return_sequences=True, input_shape = (4,5)))
    model.add(Dropout(0.2))
    model.add(GRU(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))

    es=EarlyStopping(monitor='val_loss', mode='auto', patience=50)
    rl=ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=25, factor=0.5)
    cp=ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                    filepath='./z_dacon-data/modelcheckpoint/dacon_day_2_{epoch:02d}-{val_loss:.4f}.hdf5')
    model.compile(loss=lambda x_train, y_train:quantile_loss(q, x_train, y_train), optimizer='adam')
    hist=model.fit(x_train, y_train, validation_data=(x_val,y_val),epochs=1000, batch_size=16, callbacks=[es, rl])
    loss=model.evaluate(x_test, y_test)
    pred=model.predict(x_pred)
    pred = np.where(pred < 0.4, 0, pred)
    pred = np.round_(pred,3)
    y_pred=pd.DataFrame(pred)

    file_path='./z_dacon-data/test_test/quantile_all_loss_' + str(q) + '.csv'
    y_pred.to_csv(file_path)

'''