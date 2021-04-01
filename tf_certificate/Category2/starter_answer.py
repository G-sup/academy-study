# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D,Flatten , MaxPooling1D,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    
    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)


    model = Sequential()
    model.add(Conv1D(128,4,padding='same',strides=2,input_shape=(28,28)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(Conv1D(128, 3,padding='same'))
    model.add(Dropout(0.4))
    model.add(Conv1D(64, 3))
    model.add(Dropout(0.3))
    model.add(Conv1D(64, 3))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Dense(10,activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    lr = ReduceLROnPlateau(monitor='val_loss',patience=15,factor=0.1,verbose=1) 
    es = EarlyStopping(monitor='val_loss',patience=30,mode='auto')
    model.fit(x_train,y_train, epochs = 100, batch_size = 16, validation_split = 0.2, verbose = 1,callbacks = [es,lr])
    
    print(model.evaluate(x_test,y_test))
    
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

# [0.39396727085113525, 0.8611999750137329]