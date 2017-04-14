import caffe
import numpy as np
from create_dataset_to_h5 import get_vocab
from PIL import Image

def predict_single_word(net,previous_word,image,output='probs'):
	clip = 0 if previous_word==1 else 1
	word = np.array([previous_word])
	clip = np.array([clip])
	net.forward(data=image,clip=clip,input_sentence=word)
	preds = net.blobs[output].data[0,0,:]
	return preds

def print_word(x,vocab):
	for i in x[:10]:
		print vocab[i]

vocab = get_vocab()
vinv = {}
for i,w in enumerate(vocab):
	vinv[vocab[w]]=w
img_file = '/home/aahansingh/ConsciousAgent/CaffeModel/Flicker8k_Dataset/2436081047_bca044c1d3.jpg'
img = Image.open(img_file)
img = img.resize((224,224),Image.NEAREST)
img = np.array(img)
img = np.rollaxis(img,2)
img = img[np.newaxis,:,:,:]
net = caffe.Net('LRCN.deploy.prototxt','LRCNModel_iter_50000.caffemodel', caffe.TEST)
net.blobs['data'].reshape(*img.shape)

word = 1
for i in range(40):
	print "INPUT WORD = %s" % vinv[word]
	if i ==1:
		first = predict_single_word(net,1,img)
	else:
		first = predict_single_word(net,word,img)
	first = np.argsort(-1*first)
	print first[:10]
	print_word(first,vinv)
	print '\n'
	word = first[0]


'''
print '\n'
first = predict_single_word(net,3747,img)
first = np.argsort(-1*first)
print first[:10]
print_word(first,vinv)
print '\n'
print vinv[3747]

print '\n'
first = predict_single_word(net,4113,img)
first = np.argsort(-1*first)
print first[:10]
print_word(first,vinv)
print '\n'
print vinv[4113]

print '\n'
first = predict_single_word(net,3747,img)
first = np.argsort(-1*first)
print first[:10]
print_word(first,vinv)
print '\n'
print vinv[3747]

print '\n'
first = predict_single_word(net,787,img)
first = np.argsort(-1*first)
print first[:10]
print_word(first,vinv)
print '\n'
print vinv[787]

print '\n'
first = predict_single_word(net,3197,img)
first = np.argsort(-1*first)
print first[:10]
print_word(first,vinv)
print '\n'
print vinv[3197]

print '\n'
first = predict_single_word(net,4113,img)
first = np.argsort(-1*first)
print first[:10]
print_word(first,vinv)
print '\n'
print vinv[4113]

print '\n'
first = predict_single_word(net,6304,img)
first = np.argsort(-1*first)
print first[:10]
print_word(first,vinv)
print '\n'
print vinv[6304]

print '\n'
first = predict_single_word(net,1340,img)
first = np.argsort(-1*first)
print first[:10]
print_word(first,vinv)
print '\n'
print vinv[1340]'''
