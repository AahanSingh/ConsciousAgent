from keras.models import load_model, Sequential
from keras.preprocessing import sequence
from keras.preprocessing import image
import numpy as np
import create_model
import h5py


def loadCaps(index,partial_caps, max_cap_len, first=False):
    '''

    :param index: THE INDEX OF NEXT MOST PROBABLE WORD
    :param partial_caps: LIST CONTAINING INDICES OF THE OUTPUT CAPTION
    :param max_cap_len: MAXIMUM LENGTH OF CAPTION
    :param first: IF TRUE THEN IT MEANS THAT THE PARTIAL_CAPS LIST CONTAINS ONLY 1 ELEMENT I.E. START OF PROCESS
    :return: RETURNS THE PARTIAL CAPTION PADDED TO INCREASE LENGTH TO MAX_CAP_LEN
    '''
    if(not(first)):
        partial_caps[0].append(index)
    padded_caps = sequence.pad_sequences(partial_caps, maxlen=max_cap_len, padding='post')
    return partial_caps, padded_caps

def find_max(preds):
    return np.argmax(preds)

def load_model_test1():
    model = create_model.ImageCaptioner().create_model(True)
    model.load_weights('Models/Weights.h5', by_name=True)
    print 'MODEL LOAD TEST 1 SUCCESSFUL'
    return model

# LOAD MODEL
#model = create_model.start_training()
#model = load_model('Models/model.hdf5')
model = load_model_test1()
# GENERATE TEST PARTIAL CAPS
max_cap_len = 10
partial_caps = [[1]]
partial_caps, testcaps = loadCaps(None, partial_caps, max_cap_len, True)
# LOAD TEST IMAGE
img = 'Pics/img1.jpg'
images = []
i = image.load_img(img, target_size=(224,224))
x = image.img_to_array(i)
images.append(x)
images = np.asarray(images)
preds = model.predict([images, testcaps], verbose=1)
# FIND INDEX OF MOST PROBABLE WORD
index = find_max(preds[0])
word = create_model.ImageCaptioner().get_word(index)
caption = word
print "FIRST WORD:"+word

j = 2
while(True):
    if(j ==10):
        break
    if(word == '<END>'):
        break
    partial_caps, testcaps = loadCaps(index, partial_caps, max_cap_len)
    preds = model.predict([images, testcaps], verbose=1)
    index = find_max(preds[0])
    word = create_model.ImageCaptioner().get_caps(index)
    print word
    print partial_caps
    j+=1