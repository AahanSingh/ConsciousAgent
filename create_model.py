import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding, GRU, TimeDistributed, Dense, RepeatVector, Merge, Activation, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing import image, sequence
from keras.applications.vgg16 import preprocess_input, decode_predictions


def createVGG16():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('VGG16.h5')
    return model

def createModel(vocab_size, max_caption_len, ret_model = False):
    # LOADING THE VGG16 MODEL PRETRAINED ON IMAGENET
    modelV = createVGG16()
    modelV.trainable = False
    # DISCARD LAST 2 LAYERS
    modelV.layers.pop()
    modelV.layers.pop()

    print 'LOADED VISION MODULE'

    modelL = Sequential()
    # CONVERTING THE INPUT PARTIAL CAPTION INDEX VECTOR TO DENSE VECTOR REPRESENTATION
    modelL.add(Embedding(vocab_size, 256, input_length=max_caption_len))
    modelL.add(LSTM(128,return_sequences=True))
    modelL.add(TimeDistributed(Dense(128)))

    print 'LOADED LANGUAGE MODULE'

    # REPEATING IMAGE VECTOR TO TURN INTO A SEQUENCE
    modelV.add(RepeatVector(max_caption_len))

    print 'LOADED REPEAT MODULE'

    model = Sequential()
    model.add(Merge([modelV, modelL], mode='concat', concat_axis=-1))
    # ENCODING THE VECTOR SEQ INTO A SINGLE VECTOR
    # WHICH WILL BE USED TO COMPUTE THE PROB DISTRIB OF THE NEXT WORD
    # IN THE CAPTION
    model.add(LSTM(256,return_sequences=False))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    if(ret_model==True):
        return model

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print 'COMBINED MODULES'
    # OUTPUT WILL BE OF SHAPE (samples, max_caption_len, 128)
    return model

def loadImgs():

    df = pd.read_csv('training.txt', delimiter='\t')
    nb_samples = df.shape[0]
    iter = df.iterrows()
    imgs = []
    # ADDING CAPTIONS TO THE CAPS VECTOR
    for i in range(nb_samples):
        # X[1][1] IS THE CAPTION STRING
        x = iter.next()
        imgs.append('Flicker8k_Dataset/'+x[1][0])
    #imgs=['Pics/img1.jpg','Pics/img2.jpg','Pics/img3.jpg','Pics/img4.jpg']
    images = []
    count =0
    for i in imgs:
        print 'Loaded %d' %count
        img = image.load_img(i, target_size=(224,224))
        x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        #x = x[0]
        images.append(x)
        count+=1

    images = np.asarray(images)
    return images

def loadCaps(max_cap_len, vocab_size, only_caps=True):
    '''

    :param max_cap_len: MAXIMUM LENGTH OF THE CAPTION STRING
    :param vocab_size: SIZE OF THE VOCAB
    :param only_caps: IF TRUE RETURNS ONLY PARTIAL CAPS AND NEXT WORDS. IF FALSE RETURNS THE WORD->INDEX & INDEX-> WORD DICT
    :return:
    '''
    # SAVE THE CAPTION FILE NAMED AS 'training.txt' TAB SEPARATED WITHOUT QUOTES
    df = pd.read_csv('training.txt', delimiter='\t')
    nb_samples = df.shape[0]
    iter = df.iterrows()
    caps = []
    # ADDING CAPTIONS TO THE CAPS VECTOR
    for i in range(nb_samples):
        # X[1][1] IS THE CAPTION STRING
        x = iter.next()
        caps.append(x[1][1])

    '''
    THIS WAS ONLY FOR TESTING
    caps = ['<START> An artist playing the violin <END>', '<START> A motorcycle situated along the highway <END>',
            '<START> A cowboy riding his horse in the desert . <END>', '<START> Tourist taking pictures <END>']
    '''
    words = [txt.split() for txt in caps]
    unique = []
    for word in words:
        unique.extend(word)

    unique = list(set(unique))
    # DICTIONARY OF WORDS VS INDICES OF WORDS.
    word_index = {}
    index_word = {}
    for i, word in enumerate(unique):
        word_index[word]=i
        index_word[i]=word

    partial_caps = []
    for text in caps:
        one = [word_index[txt] for txt in text.split()] # STORING THE INDEX OF THE WORD AS THEY APPEAR IN THE CAPTION STRING
        partial_caps.append(one)

    # FINDING MAX LENGTH CAPTION
    max = 0
    for i in partial_caps:
        if(len(i) > max):
            max = len(i)
    max_cap_len = max
    vocab_size = len(unique)
    # TRANSFORMING PARTIAL_CAPS INTO A SEQUENCE OF TIMESTEPS
    partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_cap_len, padding='post')
    print partial_caps

    next_words = np.zeros((nb_samples,vocab_size))
    # NEXT WORDS IS A BINARY VECTOR WHERE 1 MEANS THE WORD CORRESPONDING TO THAT INDEX IS PRESENT IN THE CAPTION
    for i,text in enumerate(caps):
        text = text.split()
        x = [word_index[txt] for txt in text]
        x = np.asarray(x)
        next_words[i,x] = 1

    if(only_caps):
        return partial_caps, next_words, max_cap_len, vocab_size
    else:
        return partial_caps,next_words,word_index,index_word

def get_caps(max_cap_len, vocab_size, index):
    partial_caps, next_words, word_index, index_word = loadCaps(max_cap_len, vocab_size, False)
    return index_word[index]


#get_caps(10,43)
def start_training():
    vocab_size = 43
    max_cap_len = 10

    model = createModel(vocab_size, max_cap_len)

    print 'LOADED COMBINED MODULE'

    images = loadImgs()
    print 'LOADED IMAGES'
    partial_caps, next_words = loadCaps(max_cap_len,vocab_size)
    print 'LOADED CAPTIONS'
    print 'STARTING TRAINING'
    model.fit([images, partial_caps], next_words, batch_size=1, nb_epoch=10, verbose=2)
    print 'TRAINING COMPLETE'
    model.save('Models/WholeModel.hdf5', overwrite=True)
    model.save_weights('Models/Weights.h5',overwrite=True)

    return model

# LOADING IMAGE
#img_path = 'test.jpg'
#img = image.load_img(img_path, target_size=(224,224))
#x = image.img_to_array(img)
#print x.shape
#x = np.expand_dims(x, axis = 0)
#print x.shape
#x = preprocess_input(x)
#print x.shape
#x = x[0]
#print x.shape

#model = createVGG16()
#preds = model.predict(x)
#print decode_predictions(preds)