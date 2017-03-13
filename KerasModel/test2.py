from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = mnist.load_data()

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 9
np.random.seed(seed)
pix = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0],1,28,28).astype('float32')
x_test = x_test.reshape(x_test.shape[0],1,28,28).astype('float32')

x_train = x_train/255
x_test = x_test/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
classes = y_test.shape[1]

def model1():
    model = Sequential()
    # 5X5 FILTER WINDOW IS USED AND TOTAL OF 32 FILTERS ARE USED. ONLY WHERE FILTER COMPLETELY OVERLAPPS THE IMAGE THE OUTPUT IS TAKEN HENCE 'VALID'
    model.add(Convolution2D(32,5,5,border_mode='valid',input_shape=(1,28,28),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) #POOL SIZE IS 2X2
    model.add(Dropout(0.2)) #EXCLUDES 20% OF THE NEURONS TO PREVENT OVERFITTING
    model.add(Flatten()) #FLATTENS INTO 1D VECTOR
    model.add(Dense(classes,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = model1()

model.fit(x_train,y_train,validation_data=(x_test,y_test),nb_epoch=10,batch_size=200,verbose=1)
scores = model.evaluate(x_test,y_test,verbose=1)
print "ERROR:"
print 100-100*scores[1]