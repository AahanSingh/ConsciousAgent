import caffe
import math
import random
import numpy as np
from create_dataset_to_h5 import get_vocab
from PIL import Image

def predict_single_word(image,previous_word,output='probs'):
    image = img
    clip = 0 if previous_word==1 else 1
    word = np.array([previous_word])
    clip = np.array([clip])
    net.forward(data=image,clip=clip,input_sentence=word)
    preds = net.blobs[output].data[0,0,:]
    return preds

def predict_word_based_on_all_previous_words(previous_words,net,image):
    preds=0
    for word in previous_words:
        preds=predict_single_word(net,word,image)
    return preds



def predict_caption(descriptor, strategy={'type': 'temp'}):

#    if strategy['type'] == 'beam':
 #     return predict_caption_beam_search(descriptor, strategy)

    num_samples = 1 #strategy['num'] if 'num' in strategy else 1
    samples = []
    sample_probs = []
    for _ in range(num_samples): # NUMBER OF CAPTIONS TO GENERATE
      sample, sample_prob = sample_caption(descriptor, strategy)
      samples.append(sample)
      sample_probs.append(sample_prob)
    return samples, sample_probs

def sample_caption(descriptor, strategy, net_output='predict', max_length=50):
    sentence = []
    probs = []
    eps_prob = 1e-8
    temp = 1 #strategy['temp'] if 'temp' in strategy else 1.0
    if max_length < 0: max_length = float('inf')
    while len(sentence) < max_length and (not sentence or sentence[-1] != 0):
      previous_word = sentence[-1] if sentence else 1
      softmax_inputs = predict_single_word(descriptor, previous_word, output=net_output)
      word = random_choice_from_probs(softmax_inputs, temp)
      sentence.append(word)
      probs.append(softmax(softmax_inputs, 1.0)[word])
    return sentence, probs

def random_choice_from_probs(softmax_inputs, temp=1, already_softmaxed=False):
  # temperature of infinity == take the max
  if temp == float('inf'):
    return np.argmax(softmax_inputs)
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?

def softmax(softmax_inputs, temp):
  shifted_inputs = softmax_inputs - softmax_inputs.max()
  exp_outputs = np.exp(temp * shifted_inputs)
  exp_outputs_sum = exp_outputs.sum()
  if math.isnan(exp_outputs_sum):
    return exp_outputs * float('nan')
  assert exp_outputs_sum > 0
  if math.isinf(exp_outputs_sum):
    return np.zeros_like(exp_outputs)
  eps_sum = 1e-20
  return exp_outputs / max(exp_outputs_sum, eps_sum)

'''def predict_caption_beam_search(descriptor, strategy, max_length=50):
    orig_batch_size = set_caption_batch_size()
    if orig_batch_size != 1: set_caption_batch_size(1)
    beam_size = strategy['beam_size'] if 'beam_size' in strategy else 1
    assert beam_size >= 1
    beams = [[]]
    beams_complete = 0
    beam_probs = [[]]
    beam_log_probs = [0.]
    while beams_complete < len(beams):
      expansions = []
      for beam_index, beam_log_prob, beam in \
          zip(range(len(beams)), beam_log_probs, beams):
        if beam:
          previous_word = beam[-1]
          if len(beam) >= max_length or previous_word == 0:
            exp = {'prefix_beam_index': beam_index, 'extension': [],
                   'prob_extension': [], 'log_prob': beam_log_prob}
            expansions.append(exp)
            # Don't expand this beam; it was already ended with an EOS,
            # or is the max length.
            continue
        else:
          previous_word = 0  # EOS is first word
        if beam_size == 1:
          probs = predict_single_word(descriptor, previous_word)
        else:
          probs = predict_single_word_from_all_previous(descriptor, beam)
        assert len(probs.shape) == 1
        assert probs.shape[0] == len(vocab)
        expansion_inds = probs.argsort()[-beam_size:]
        for ind in expansion_inds:
          prob = probs[ind]
          extended_beam_log_prob = beam_log_prob + math.log(prob)
          exp = {'prefix_beam_index': beam_index, 'extension': [ind],
                 'prob_extension': [prob], 'log_prob': extended_beam_log_prob}
          expansions.append(exp)
      # Sort expansions in decreasing order of probability.
      expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
      expansions = expansions[:beam_size]
      new_beams = \
          [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
      new_beam_probs = \
          [beam_probs[e['prefix_beam_index']] + e['prob_extension'] for e in expansions]
      beam_log_probs = [e['log_prob'] for e in expansions]
      beams_complete = 0
      for beam in new_beams:
        if beam[-1] == 0 or len(beam) >= max_length: beams_complete += 1
      beams, beam_probs = new_beams, new_beam_probs
    if orig_batch_size != 1: set_caption_batch_size(orig_batch_size)
    return beams, beam_probs
'''
def set_caption_batch_size(batch_size,lstm_net):
    lstm_net.blobs['cont_sentence'].reshape(1, batch_size)
    lstm_net.blobs['input_sentence'].reshape(1, batch_size)
    lstm_net.blobs['image_features'].reshape(batch_size, lstm_net.blobs['image_features'].data.shape[1:])
    lstm_net.reshape()

def gen_caption(vocab):
    '''
    THIS FUNCTION WILL GENERATE CAPTIONS.
    IT WILL USE THE TOP 5 MOST LIKELY WORDS AND RUN A* TO FIND THE BEST CAPTION.
    :param vocab:
    :return:
    '''

    vocab = get_vocab()
    vinv = {}
    # OBTAIN DICTIONARY
    for i, w in enumerate(vocab):
        vinv[vocab[w]] = w
    # OBTAIN IMAGE
    img_file = '/home/aahansingh/ConsciousAgent/CaffeModel/Flicker8k_Dataset/2436081047_bca044c1d3.jpg'
    img = Image.open(img_file)
    img = img.resize((224, 224), Image.NEAREST)
    img = np.array(img)
    img = np.rollaxis(img, 2)
    img = img[np.newaxis, :, :, :]
    net = caffe.Net('LRCN.deploy.prototxt', 'LRCNModel_iter_50000.caffemodel', caffe.TEST)
    net.blobs['data'].reshape(*img.shape)

    word = 1 # 1==<START>
    preds = predict_single_word(net,word,img)
    top_5_words = np.argsort(-preds) # HOLDS THE INDICES OF THE TOP 5 WORDS
    top_5_words = top_5_words[:5]
    top_5_preds=np.zeros(5) # HOLDS THE PROBABILITIES OF TOP 5 WORDS
    for i,j in enumerate(top_5_words):
        top_5_preds[i]=preds[j]



    for i in range(40):
        print "INPUT WORD = %s" % vinv[word]
        if i == 1:
            first = predict_single_word(net, 1, img)
        else:
            first = predict_single_word(net, word, img)
        first = np.argsort(-first)  # ARGSORT SORTS IN ASCENDING. HERE WE SORT IN DESCENDING
        print first[:10]
        print_word(first, vinv)
        print '\n'
        word = first[0]


def print_word(x,vocab):
    for i in x:
        print vocab[i]

def process_img(img_file):
    img = Image.open(img_file)
    img = img.resize((224, 224), Image.NEAREST)
    img = np.array(img)
    img = np.rollaxis(img, 2)
    img = img[np.newaxis, :, :, :]
    return img

def gen_cap():
	word = 1	
	cum_sum=0.
	preds = 0
	sent = ""	
	for _ in range(100):
		if word==0:
			break
		print vinv[word]			
		#word = raw_input("ENTER INDEX")
		#word = int(word)
		if word!=1:
			cum_sum+= preds[word]		
		preds = predict_single_word(img,word,output='probs')
		top_10 = np.argsort(-preds)[:10] # HOLDS THE TOP 10 INDICES
		#print top_10		
		word = top_10[0]				
		#print_word(top_10,vinv)
		sent+=" "+vinv[word]
	print cum_sum
import os
vocab = get_vocab()
vinv = {}
# OBTAIN DICTIONARY
for i, w in enumerate(vocab):
    vinv[vocab[w]] = w
# OBTAIN IMAGE
img_file = os.getcwd()+'/girl.jpeg'
img = process_img(img_file)

net = caffe.Net('LRCN.deploy.prototxt', 'LRCNModel_iter_60000.caffemodel', caffe.TEST)
net.blobs['data'].reshape(*img.shape)

gen_cap()
'''caption,probs = predict_caption(1223,'not_')
print caption
print vinv[1]
print_word(caption[0],vinv)
'''
