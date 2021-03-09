import pandas as pd
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

wine = pd.read_csv('C:/data/Csv/winequality-white.csv', sep = ";", header=0)

count_data = wine.groupby('quality')['quality'].count()
print(count_data)

import matplotlib.pyplot as plt

count_data.plot()
plt.show()