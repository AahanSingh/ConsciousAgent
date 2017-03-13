import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dropout, Input
from keras.preprocessing import image, sequence
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot

class ImageCaptioner():

    def __init__(self):
        self.max_cap_len = 0
        self.vocab_size = 0
        self.partial_caps = []
        self.next_words = []
        self.index_word = {}
        self.word_index = {}
        self.imgs = []

    def loadCaps(self, get_max_len = False):
        '''
        THIS FUNCTION IS USED TO LOAD CAPTIONS
        :SETS:
                 partial_caps-> PARTIAL CAPTION MATRIX
                 next_words->   NEXT WORDS MATRIX
                 max_cap_len->  MAX CAPTION LENGTH
                 vocab_size->   VOCABULARY SIZE
                 word_index->   WORD TO INDEX DICTIONARY
                 index_word->   INDEX TO WORD DICTIONARY

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




        #UNCOMMENT TO TEST
        caps = ['<START> An artist playing the violin <END>', '<START> A motorcycle situated along the highway <END>',
                '<START> A cowboy riding his horse in the desert . <END>', '<START> Tourist taking pictures <END>']
        nb_samples = 4


        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        # DICTIONARY OF WORDS VS INDICES OF WORDS.
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word]=i
            self.index_word[i]=word

        self.partial_caps = []
        for text in caps:
            one = [self.word_index[txt] for txt in text.split()] # STORING THE INDEX OF THE WORD AS THEY APPEAR IN THE CAPTION STRING
            self.partial_caps.append(one)

        # FINDING MAX LENGTH CAPTION
        max = 0
        for i in self.partial_caps:
            if(len(i) > max):
                max = len(i)
        self.max_cap_len = max
        self.vocab_size = len(unique)
        if(get_max_len):
            return
        # TRANSFORMING PARTIAL_CAPS INTO A SEQUENCE OF TIME-STEPS
        self.partial_caps = sequence.pad_sequences(self.partial_caps, maxlen=self.max_cap_len, padding='post')
        #print partial_caps

        self.next_words = np.zeros((nb_samples,self.vocab_size))
        # NEXT WORDS IS A BINARY VECTOR WHERE 1 MEANS THE WORD CORRESPONDING TO THAT INDEX IS PRESENT IN THE CAPTION
        for i,text in enumerate(caps):
            text = text.split()
            x = [self.word_index[txt] for txt in text]
            x = np.asarray(x)
            self.next_words[i,x] = 1

    def get_img_locn(self):
        '''
        THIS FUNCTION IS USED TO GET THE IMAGE LOCATION VECTOR
        '''
        # UNCOMMENT FOR TESTING
        self.imgs = ['Pics/img1.jpg', 'Pics/img2.jpg', 'Pics/img3.jpg', 'Pics/img4.jpg']
        return

        df = pd.read_csv('training.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        self.imgs = []
        # ADDING CAPTIONS TO THE CAPS VECTOR
        for i in range(nb_samples):
            # X[1][1] IS THE CAPTION STRING
            x = iter.next()
            self.imgs.append('Flicker8k_Dataset/' + x[1][0])

    def loadImgs(self, l, u):
        '''
        THIS FUNCTION IS USED TO OBTAIN THE IMAGE MATRIX
        :param l: LOWER LIMIT
        :param u: UPPER LIMIT
        :param imgs: IMAGE LOCATIONS
        :return: IMAGES
        '''
        # UNCOMMENT FOR TESTING
        l=0
        u=4

        self.get_img_locn()
        temp_imgs = self.imgs[l:u]
        images = []
        count =0
        for i in temp_imgs:
            print 'LOADED %d' %count
            img = image.load_img(i, target_size=(224,224))
            x = image.img_to_array(img)
            images.append(x)
            count+=1
        images = np.asarray(images)
        return images



    def createVGG16(self, first=False):
        '''
        THIS FUNCTION CREATES THE VGG16 MODEL
        :param first: ONLY TRUE FOR THE FIRST TIME TO SAVE VGG16 WEIGHTS
        :return: CNN MODEL
        '''
        if(first):
            model = VGG16()
            model.save_weights('VGG16.h5')

        model = Sequential()
        #model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(224,224,3), trainable=False))
        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        #model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

        model.add(Flatten(trainable=False))
        model.add(Dense(4096, activation='relu', trainable=False))
       # model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', trainable=True))
       # model.add(Dropout(0.5))
        #model.add(Dense(1000, activation='softmax'))

        model.load_weights('VGG16.h5', by_name=True)
        model.layers.pop()
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        plot(model, to_file='model2.png')
        return model

    def create_model(self, ret_model = False):
        '''

        :param vocab_size: VOCABULARY SIZE
        :param max_caption_len: MAXIMUM CAPTION LENGTH
        :param ret_model: SET TO TRUE ONLY TO CREATE MODEL WHILE TESTING
        :return:
        '''
        # LOADING THE VGG16 MODEL PRETRAINED ON IMAGENET

        modelV = self.createVGG16()
        modelV.add(RepeatVector(self.max_cap_len))

        print 'LOADED VISION MODULE'

        modelL = Sequential()
        # CONVERTING THE INPUT PARTIAL CAPTION INDEX VECTOR TO DENSE VECTOR REPRESENTATION
        modelL.add(Embedding(self.vocab_size, 1000, input_length=self.max_cap_len))
        #modelL.add(LSTM(1000,return_sequences=True))
        #modelL.add(TimeDistributed(Dense(256)))

        print 'LOADED LANGUAGE MODULE'

        # REPEATING IMAGE VECTOR TO TURN INTO A SEQUENCE


        print 'LOADED REPEAT MODULE'

        model = Sequential()
        model.add(Merge([modelV, modelL], mode='concat'))
        # ENCODING THE VECTOR SEQ INTO A SINGLE VECTOR
        # WHICH WILL BE USED TO COMPUTE THE PROB DISTRIB OF THE NEXT WORD
        # IN THE CAPTION
        model.add(LSTM(1000,return_sequences=False))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))

        if(ret_model==True):
            return model

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        print 'COMBINED MODULES'
        # OUTPUT WILL BE OF SHAPE (samples, max_caption_len, 128)
        return model

    def get_word(self,index):
        '''
        THIS FUNCTION IS USED DURING PREDICTION. RETURNS THE WORD CORRESPONDING TO INDEX
        '''
        return self.index_word[index]

    def start_training(self):

        print 'LOADING CAPTIONS'
        self.loadCaps()
        print 'LOADED CAPTIONS'

        print 'LOADING CAPTIONING MODEL'
        model = self.create_model()
        print 'LOADED CAPTIONING MODEL'

        print 'LOADING IMAGES'
        images = self.loadImgs(0,4)
        print 'LOADED IMAGES'

        print images.shape
        print self.partial_caps.shape
        file = 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
        checkpoint = ModelCheckpoint(file, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        print 'STARTING TRAINING'
        model.fit([images, self.partial_caps], self.next_words, batch_size=1, nb_epoch=20, verbose=2, callbacks=callbacks_list)
        print 'TRAINING COMPLETE'
        model.save('Models/WholeModel.hdf5', overwrite=True)
        model.save_weights('Models/Weights.h5',overwrite=True)
        print 'MODEL SAVED'

obj = ImageCaptioner()
#obj.createVGG16()
obj.start_training()
