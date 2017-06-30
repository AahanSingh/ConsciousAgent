import caffe
import math
import random
import numpy as np
from create_dataset_to_h5 import get_vocab
from PIL import Image
import time
import os

def predict_single_word(previous_word,output='probs'):#def predict_single_word(image,previous_word,output='probs'):
    image = img
    clip = 0 if previous_word==1 else 1
    word = np.array([previous_word])
    clip = np.array([clip])
    net.forward(data=image,clip=clip,input_sentence=word)
    preds = net.blobs[output].data[0,0,:]
    #print preds.shape
    return preds

def predict_word_from_all_previous_words(previous_words):#def predict_word_from_all_previous_words(previous_words,net,image):
    preds=0
    for word in previous_words:
        #print word
        #print 'IN LOOP'
        preds=predict_single_word(word)
    #print "IN ALL PREV"
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
   # assert temp == 1
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
  #assert exp_outputs_sum > 0
  if math.isinf(exp_outputs_sum):
    return np.zeros_like(exp_outputs)
  eps_sum = 1e-20
  return exp_outputs / max(exp_outputs_sum, eps_sum)

def beam_search(beam_size, max_len = 40):
	beams = [[]]
	beams_complete = 0
	beam_probs = [[]]
	word_probs = [[]]
	complete_beams=[]
	complete_beams_probs = []
	while beams_complete<len(beams):
		print "*******************************COMPLETE BEAMS = %d***************************" % beams_complete
		child_nodes = [] # HOLDS EXPANDED NODES
		for index,beam, beam_prob, word_prob in zip(range(len(beams)),beams,beam_probs,word_probs):
			print '############################## BEAM %d #######################' % index
			print beam
			# FINDING PREVIOUS WORD
			if beam: # IF BEAM EXIXTS
				prev_word = beam[-1]
				if prev_word == 0 or len(beam)>=max_len: # <END> == 0
					#node = {'parent_beam_index': index, 'word': [], 'prob_word': word_prob, 'prob_beam': beam_prob}
					#child_nodes.append(node)
					continue
			else:
				prev_word = 1 # <START> SYMBOL
				beam.append(prev_word)
			# OBTAIN PREDICTIONS
			if beam_size==1:
				preds = predict_single_word(prev_word)
			else:
				preds = predict_word_from_all_previous_words(beam)
			top_n_indices = np.argsort(-preds)[:beam_size]
			top_n_probs = preds[top_n_indices] # HOLDS CUMULATIVE PROB OF ENTIRE BEAM
			# FINDING CUMULATIVE PROB OF ENTIRE BEAM
			for i,p in enumerate(top_n_probs):
				if beam[-1]!=1:
					top_n_probs[i] = beam_prob[-1]+p 
			# CREATING NODES
			print "######################### EXPANSION INDS ############################"
			print top_n_indices
			for i,j in enumerate(top_n_indices):
				prob_b = top_n_probs[i]
				prob_w = preds[j]
				node = {'parent_beam_index': index, 'word': [j], 'prob_word': [prob_w], 'prob_beam':[prob_b]}
				child_nodes.append(node)
			print "######################### CHILD NODES IN LOOP ############################"
			print child_nodes    
		'''# SORTING CHILD NODES BY WORD PROB
		child_nodes.sort(key=lambda x:x['prob_word'],reverse=True)
		# TAKE ONLY TOP N CHILD NODES
		child_nodes = child_nodes[:beam_size]
		# SORTING CHILD NODES BY BEAM PROB'''
		child_nodes.sort(key=lambda x:x['prob_beam'],reverse=True)
		# TAKE ONLY TOP N CHILD NODES
		child_nodes = child_nodes[:beam_size]
		print "######################### SORTED CHILD NODES ############################"
		print child_nodes
		new_beams = [beams[c['parent_beam_index']] + c['word'] for c in child_nodes]
		new_beam_probs = [beam_probs[c['parent_beam_index']] + c['prob_beam'] for c in child_nodes]
		new_word_probs = [word_probs[c['parent_beam_index']] + c['prob_word'] for c in child_nodes]
		for i,beam in enumerate(new_beams):
			if beam[-1] == 0 or len(beam)>max_len:
				beams_complete+=1
				complete_beams.append(beam)
				complete_beams_probs.append(new_beam_probs[i][-1])
		beams,beam_probs, word_probs = new_beams,new_beam_probs, new_word_probs
		'''if len(new_beams) > beam_size:
			beams,beam_probs, word_probs = new_beams,new_beam_probs, new_word_probs
		else:
			beams, beam_probs, word_probs = (new_beams+beams)[:beam_size], (new_beam_probs+beam_probs)[:beam_size], (new_word_probs+word_probs)[:beam_size]
		'''
		print "######################### NEW BEAMS ############################"
		print beams
	return beams,beam_probs
	complete_beams_probs = np.array(complete_beams_probs)
	ind = np.argsort(-complete_beams_probs)
	print ind
	print complete_beams
	print complete_beams_probs
	copy = []
	for i in ind:
		copy.append(complete_beams[i])
	complete_beams_probs = complete_beams_probs[ind]
	return copy, complete_beams_probs

def save_file(filename,list):
	with open(filename, 'wb') as file:
		for word in list:
			file.write("%s\n" %word)


def create_lists(choice):
	sentient = []
	not_sentient = []
	objects = []
	non_objects = []
	verbs = []
	i = 0
	if choice == 0: #0 = objects
		for word in vocab:
			'''
			1 = object
			0 = not object
			-1 = exit
			'''
			x = raw_input("%d WORD = %s\nENTER 1:Object\t 0:Non Object\n"%(i,word))
			x = int(x)
			if x==1:
				objects.append(word)
			if x==0:
				non_objects.append(word)
			if x==-1:
				save_file('objects.txt',objects)
				save_file('non_objects.txt',non_objects)
			i+=1
		save_file('objects.txt',objects)
		save_file('non_objects.txt',non_objects)	
	if choice == 1:# VERBS
		for word in vocab:
			if 'ing' in word:
				verbs.append(word)


def print_sentence(x):
    s = ""
    for i in x:
        s+=" "+vinv[i]
    return s
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
        word = raw_input("ENTER INDEX")
        word = int(word)
        if word!=1:
            cum_sum+= preds[word]
        preds = predict_single_word(word)
        top_10 = np.argsort(-preds)[:10] # HOLDS THE TOP 10 INDICES
        print top_10
        #word = top_10[0]
        print_sentence(top_10)
        sent+=" "+vinv[word]
    print sent
    print cum_sum
def read_file(filename):
    wi = []
    with open(filename) as f:
        for line in f:
            wi.append(line)
    for x in range(len(wi)):
    	wi[x]=wi[x][:-1]
    return wi

def sentience_engine(sentence):
	# sentence is the vector of ints
	sentence = list(set(sentence))
	sentient = read_file('/home/ubuntu/ConsciousAgent/sentient.txt')
	non_sentient = read_file('/home/ubuntu/ConsciousAgent/non_sentient.txt')
	fish = read_file('/home/ubuntu/ConsciousAgent/fish.txt')
	human=read_file('/home/ubuntu/ConsciousAgent/human.txt')
	human_groups=read_file('/home/ubuntu/ConsciousAgent/human_groups.txt')
	plants = read_file('/home/ubuntu/ConsciousAgent/plants.txt')
	plant_group = read_file('/home/ubuntu/ConsciousAgent/plant_group.txt')
	animals = read_file('/home/ubuntu/ConsciousAgent/animals.txt')
	animal_group = read_file('/home/ubuntu/ConsciousAgent/animal_group.txt')
	motion = read_file('/home/ubuntu/ConsciousAgent/motion.txt')
	static = read_file('/home/ubuntu/ConsciousAgent/static.txt')
	fantasy = read_file('/home/ubuntu/ConsciousAgent/fantasy.txt')

	# SENTIENCE DETECTION
	for word in sentence:
		result = ""
		if word in sentient:
			result+="Sentient Object Detected => "
			if word in human:
				result+="Human => "
				if word in human_groups:
					result+=" Group => "
				result+=word
				print result
				file2.write(result+'\n')
				continue
			if word in fish:
				result+="Fish => "+word
				print result
				file2.write(result+'\n')
				continue
			if word in plants:
				result+="Plant => "
				if word in plant_group:
					result+="Group => "
				result+=word
				print result
				file2.write(result+'\n')
				continue
			if word in animals:
				result+="Animal => "
				if word in animal_group:
					result+="Group => "
				result+=word
				print result
				file2.write(result+'\n')
				continue
			if word in fantasy:
				result+="Fantasy Creature => "+word
				print result
				file2.write(result+'\n')
				continue
		if word in non_sentient:
			result+="Non Sentient Object Detected => "+word
			print result
			file2.write(result+'\n')
			continue
	# MOTION DETECTION
	'beach' in motion
	for word in sentence:
		result = ""
		if word in motion:
			result+='Motion Detected => '+word
			print result
			file2.write(result+'\n')
			continue
		if word in static:
			result+='Objects Stationary => '+word
			print result
			file2.write(result+'\n')
			continue

start_time = time.time()
net = caffe.Net('LRCN.deploy.prototxt', 'LRCN_Custom_dataset_Model_iter_240.caffemodel', caffe.TEST)
net.blobs['data'].reshape(*img.shape)

vocab = get_vocab()
vinv = {}
# OBTAIN DICTIONARY
for i, w in enumerate(vocab):
    vinv[vocab[w]] = w

file1 = open('output.txt','a+')
file2 = open('output_complete.txt','a+')

for i in range(1,50):
	# OBTAIN IMAGE
	#image_locn = raw_input("Input Image Name: ")
	image_locn='TEST/test%d.jpg' % i
	img_file = os.getcwd()+"/"+image_locn
	img = process_img(img_file)

	#create_lists(0)
	#gen_cap()

	#beam_size = int(raw_input("ENTER BEAM SIZE: "))
	beam_size=5
	beams,probs = beam_search(beam_size)
	#print beams.shape
	#print probs.shape
	'''
	'''
	
	n_caps = []
	file1.write('\n'+image_locn+'\n')
	file2.write('\n'+image_locn+'\n')
	for i in beams:
	    for j in i:
	        print vinv[j]
	'''
	'''
	print '*****************************************************************************************'
	print 'Top %d Caption Probabilities: ' %beam_size
	file1.write('Top %d Caption Probabilities: \n' %beam_size)
	for i in range(beam_size):
		print probs[i][-1]
		file1.write(str(probs[i][-1])+'\n')
	
	print '\nTop %d Captions: '%beam_size
	for sent in beams:
		temp=print_sentence(sent)+"\n"
		n_caps.append(temp)

	for i in n_caps:
		file1.write(i)

	#print probs[0][-1]
	print '\nBest Caption: '
	best_cap = print_sentence(beams[0])+'\n'
	file2.write(best_cap)
	beam = []
	for i in beams[0]:
		beam.append(vinv[i])
	#os.system("play -n synth 2 sine 800 vol 0.5")
	print '\nSentience Engine Running'
	file2.write('\nSentience Engine Running')
	sentience_engine(beam)
	print '\nSentience Detection Complete'
	file2.write('\nSentience Engine Complete')
	print '\nTotal Time Taken'
	print time.time()-start_time

file1.close()
file2.close()