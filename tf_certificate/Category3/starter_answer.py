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
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


from re import VERBOSE
import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    # YOUR CODE HERE
    train_generator = training_datagen.flow_from_directory(directory = TRAINING_DIR, target_size=(150, 150), batch_size=20,subset='training')

    test_generator = training_datagen.flow_from_directory(directory = TRAINING_DIR, target_size=(150, 150), batch_size=20, subset='validation')

    

    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(32,3,input_shape = (150,150,3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(32,3),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(32,3),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(32,3),
        tf.keras.layers.MaxPooling2D(3),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(16,3),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(16,3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics='acc')
    lr = ReduceLROnPlateau(monitor='val_loss',patience=15,factor=0.1,verbose=1) 
    es = EarlyStopping(monitor='val_loss',patience=30,mode='auto')
    hist = model.fit(train_generator, steps_per_epoch=len(train_generator),epochs=1000,validation_data=test_generator,validation_steps=len(test_generator),callbacks=[es,lr])


    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    print(loss[-1])
    print(val_loss[-1])
    print(acc[-1])
    print(val_acc[-1])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

# 0.0008487657178193331
# 0.0009952995460480452
# 0.9995039701461792
# 1.0