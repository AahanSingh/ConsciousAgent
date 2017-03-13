from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing import image
import numpy as np
import create_model
import h5py


def loadCaps(partial_caps, max_cap_len):
    '''
    :param partial_caps: LIST CONTAINING INDICES OF THE OUTPUT CAPTION
    :param max_cap_len: MAXIMUM LENGTH OF CAPTION
    :return: RETURNS THE PARTIAL CAPTION PADDED TO INCREASE LENGTH TO MAX_CAP_LEN
    '''
    partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_cap_len, padding='post')
    return partial_caps

def find_max(preds):
    m = max(preds)
    ind = preds.index(m)
    return ind

def generate_caption(max_cap_len, vocab_size, images):

    partial_caps = [[1]]
    testcaps = loadCaps(partial_caps, max_cap_len)
    print testcaps.shape
    # START PREDICTION
    print 'STARTING PREDICTION'
    caption = ""
    while(True):

        preds = model.predict([images, testcaps], verbose=1)
        # FIND INDEX OF MOST PROBABLE WORD
        index = find_max(preds[0])
        word = create_model.get_caps(max_cap_len, vocab_size, index)
        print word
        if(word == '<END>'):
            break
        else:
            partial_caps[0].append(index)
            caption+=word
            testcaps = loadCaps(partial_caps, max_cap_len)
    return caption
        # PREDICT AGAIN UNTIL THE INDEX OF END OF OBTAINED


# LOAD MODEL
model = create_model.start_training()

# GENERATE TEST PARTIAL CAPS
max_cap_len = 10

# LOAD TEST IMAGE
img = 'Pics/img1.jpg'
images = []
i = image.load_img(img, target_size=(224,224))
x = image.img_to_array(i)
images.append(x)
images = np.asarray(images)
caption = generate_caption(max_cap_len, 43, image)