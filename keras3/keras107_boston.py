from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.datasets import load_boston
import numpy as np
import tensorflow as tf
import autokeras as ak

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape)
print(y.shape)

x_train, x_test, y_train,y_test = train_test_split(x, y, train_size = 0.8 , random_state = 104 )

model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=2, # 몇번 시도할것인가
    loss='mse',
    metrics=['mae']
)

model.fit(x_train,y_train, epochs = 10, validation_split = 0.2)

results = model.evaluate(x_test, y_test)
print(results)

model2 = model.export_model()
try:
    model2.save('./ak_test/boston', save_format='tf')
except:
    model2.save('./ak_test/boston.h5')

best_model = model.tuner.get_best_model()
try:
    best_model.save('./ak_test/best_boston', save_format='tf')
except:
    best_model.save('./ak_test/best_boston.h5')

