import pandas as pd
import numpy as np
import h5py

def build_dataset():

    token_path = "/Users/aahansingh/Development/Python/ConsciousAgent1/captions.txt"
    df = pd.read_csv(token_path, delimiter = '\t')
    n_caps = df.shape[0]
    dataset = []
    iterator = df.iterrows()
    for i in range(0,n_caps,5):
        max = 0
        q = 0
        for j in range(i, i+5):
            p = iterator.next()
            len = sum(c.isspace() for c in p[1][1]) + 1
            if(len > max):
                q = p
                max = len
        dataset.append((q[1][0],q[1][1]))
    # SAVE DATASET
    with open('dataset.txt', 'wb') as file:
        for i, w in enumerate(dataset):
            file.write("%s\t%s\n" % (w[0][:-2], w[1].lower()))
    with open('train_images.txt', 'wb') as file:
        for i, w in enumerate(dataset):
            file.write("Flicker8k_Dataset/%s\n") % w[0][:-2]
    print 'DATASET SAVED'
    return dataset

def build_vocab():
    token_path = "/Users/aahansingh/Development/Python/ConsciousAgent1/captions.txt"
    df = pd.read_csv(token_path, delimiter='\t')
    n_caps = df.shape[0]
    captions = []
    # RESET ITERATOR
    iterator = df.iterrows()
    for i in range(n_caps):
        # X[1][1] IS THE CAPTION STRING
        x = iterator.next()
        captions.append(x[1][1].lower())

    words = [txt.split() for txt in captions]
    unique = []
    for word in words:
        unique.extend(word)
    unique = list(set(unique))
    unique.append("<START>")
    unique.append("<END>")
    unique = unique[::-1]
    # CREATE DICTIONARY
    word_index = {}
    index_word = {}
    for i, word in enumerate(unique):
        word_index[word] = i
        index_word[i]=word
    # SAVE VOCABULARY
    with open('vocabulary_dictionary.txt', 'wb') as file:
        for index in index_word:
            file.write("%s\t%d\n" %(index_word[index],index))
    print 'DICTIONARY SAVED'

def get_vocab():
    wi = {}
    with open('vocabulary_dictionary.txt') as f:
        for line in f:
            (word, index) = line.split('\t')
            wi[word] = int(index)
    return wi

def get_dataset():
    wi = {}
    with open('dataset.txt') as f:
        for line in f:
            (img, cap) = line.split('\t')
            wi[img] = cap[:-1]
    return wi

def build_training_set():

    print 'BUILDING DATA HDF5 FILE'
    # LOADING TRAINING DATA
    dataset = get_dataset()

    # CREATE DICTIONARY
    word_index = get_vocab()

    # IMPORT TRAIN
    with open('Flickr8k_text/Flickr_8k.trainImages.txt') as f:
        train_imgs = f.readlines()

    # OBTAIN CAPTION FOR IMAGES :=> dataset[train_imgs[0][:-1]]  -1 due to \n at the end of the
    max_cap_len = 0
    train_captions = []

    # LOADING CAPTIONS
    for i in range(len(train_imgs)):
        train_captions.append(dataset[train_imgs[i][:-1]])

    numeric_train_captions = []

    # CONVERT TO NUMERIC
    for i in range(len(train_captions)):
        caption = train_captions[i].strip().split()
        if(len(caption)>max_cap_len):
            max_cap_len = len(caption)
        temp = []
        for j in range(len(caption)):
            temp.append(word_index[caption[j]])
        numeric_train_captions.append(temp)

    max_cap_len+=2
    # PREPEND WITH <START> AND APPEND WITH <END>
    data = {'input':[],'target':[],'clip':[]}
    for i in range(len(numeric_train_captions)):
        numeric_train_captions[i] = [word_index['<START>']] + numeric_train_captions[i] + [word_index['<END>']]
        diff = max_cap_len - len(numeric_train_captions[i])
        input = numeric_train_captions[i] + [-1] * diff
        clip = [0] + [1]*(len(numeric_train_captions[i])-1) + [0]*diff
        target = numeric_train_captions[i][1:] + [-1] * (diff+1)
        data['input'].append(input)
        data['target'].append(target)
        data['clip'].append(clip)

    data['input'] = np.asarray(data['input'])
    data['target'] = np.asarray(data['target'])
    data['clip'] = np.asarray(data['clip'])

    print 'Input = (%d,%d)'%data['input'].shape
    print 'Target = (%d,%d)' % data['target'].shape
    print 'Clip = (%d,%d)' % data['clip'].shape

    print 'SAVING TO HDF5'
    file_names = []
    # WRITE TO H5 FILE. CAFFE TAKES 1 INPUT FOR HDF5 DATA LAYER:
    # THE TXT FILE CONTAINING LOCAITON OF H5 FILE
    for i in range(0,6000,100):
        file_name = 'train_captions%d.h5' % i
        file_names.append(file_name)
        with h5py.File(file_name,'w') as f:
            f['input'] = data['input'][i:i+100,:].T
            f['target'] = data['target'][i:i+100,:].T
            f['clip'] = data['clip'][i:i+100,:].T
        print 'SAVED %s' %file_name

    f = open('train_captions.txt','w')
    for filename in file_names:
        f.write('%s\n'%filename)
    f.close()
    print 'DONE'
'''
    data['input'] = data['input'].T
    data['target'] = data['target'].T
    data['clip'] = data['clip'].T

    with h5py.File('train_captions_transpose.h5','w') as f:
        f['input'] = data['input']
        f['target'] = data['target']
        f['clip'] = data['clip']
    f = open('train_captions_transpose.txt','w')
    f.write('train_captions_transpose.h5')
    f.close()'''



build_dataset()
build_vocab()
build_training_set()
