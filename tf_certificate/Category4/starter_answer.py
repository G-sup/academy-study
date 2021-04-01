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
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('sarcasm.json', 'r') as f:
        datasets = json.load(f)
    for item in datasets:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
    
    token = Tokenizer(num_words=vocab_size, oov_token= oov_tok)  

    token.fit_on_texts(sentences)

    sentences = token.texts_to_sequences(sentences)
    print(sentences)
    x_train = np.array(sentences[0:training_size])
    x_test = np.array(sentences[training_size:])
    y_train = np.array(labels[0:training_size])
    y_test = np.array(labels[training_size:])

    x_train = pad_sequences(x_train, padding=padding_type, truncating=trunc_type) 
    x_test = pad_sequences(x_test, padding=padding_type, truncating=trunc_type)


    x_train = pad_sequences(x_train, maxlen = max_length, padding='pre')
    x_test = pad_sequences(x_test, maxlen = max_length, padding='pre')

    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(input_dim = vocab_size,output_dim = embedding_dim,input_length = max_length),
        tf.keras.layers.Conv1D(32,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary
    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss',patience=15,mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss',factor = 0.1, patience= 5)
    model.fit( x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, verbose=1, callbacks=[es,lr])
    print(model.evaluate(x_test,y_test))
    
    return model


# # Note that you'll need to save your model as a .h5 like this.
# # When you press the Submit and Test button, your saved .h5 model will
# # be sent to the testing infrastructure for scoring
# # and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

# [0.468118816614151, 0.8262035846710205]