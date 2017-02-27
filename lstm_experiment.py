from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np
import sys

file = 'alice.txt'
text = open(file).read()
text = text.lower()
chars = sorted(list(set(text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
n_chars = len(text) #total no of chars
n_vocab = len(chars)#no of unique chars
seq_length = 100
dataX = []
dataY = []
for i in range(0,n_chars-seq_length,1):
    seq_in = text[i:i+seq_length]
    seq_out = text[i+seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patt = len(dataX)
# RESHAPE TO GIVE INPUT TO LSTM :- (SAMPLES, TIMESTEPS OR NUMBER OF CHARS, FEATURES)
X = np.reshape(dataX,(n_patt,seq_length,1))
# NORMALIZE 0->1
X = X/float(n_vocab)
# ONE HOT ENCODE OUTPUT VECTOR
y = np_utils.to_categorical(dataY)
# LSTM WITH 256 MEM UNITS
print X.shape
print y.shape
model = Sequential()
model.add(LSTM(256,input_dim=1, input_length=100))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

file = 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(file,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks_list = [checkpoint]
#model.fit(X,y,nb_epoch=20,batch_size=256,callbacks=callbacks_list)
model.load_weights('weights-improvement-19-2.0474.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam')
int_to_char = dict((i, c) for i, c in enumerate(chars))
# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    print prediction
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print "\nDone."