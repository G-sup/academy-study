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

df = pd.read_csv('./z_dacon-data/train/train.csv', index_col=[0,1,2], header=0) 

con = df[['DHI','DNI','RH','T','TARGET']]

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[[ 'TARGET', 'DHI', 'DNI','RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['TARGET', 'DHI', 'DNI', 'RH', 'T']]
                              
        return temp.iloc[-48:, :]


df_train = preprocess_data(con)
df_train.iloc[:48]

df_test = []

for i in range(81):
    file_path = './z_dacon-data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)
print(X_test.shape)

# X_test.to_csv('./z_dacon-data/test_all.csv', sep=',') 
