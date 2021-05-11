from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.datasets import load_breast_cancer


datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape)
print(y.shape)

x_train, x_test, y_train,y_test = train_test_split(x, y, train_size = 0.8 , random_state = 104 )

model = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=2, # 몇번 시도할것인가
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train, epochs = 10, validation_split = 0.2)

results = model.evaluate(x_test, y_test)
print(results)

model2 = model.export_model()
model2.summary()
model2.save('./ak_test/cancer.h5')

best_model = model.tuner.get_best_model()
best_model.save('./ak_test/best_cancer.h5')
