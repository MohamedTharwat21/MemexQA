import os
import json
import nltk
import random
import numpy as np
from tqdm import tqdm
import pickle
import argparse
from html.parser import HTMLParser
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import LongformerTokenizer, LongformerModel
# from configuration import config

class config:
    def __init__(self):
        self.albumjson=r'E:\Memex_QA\memexqa_dataset_v1.1\album_info.json'   #"path to album_info.json"
        self.qas=r'E:\Memex_QA\memexqa_dataset_v1.1\qas.json' #"path to the qas.json"
        self.glove= r'E:\Memex_QA\memexqa_dataset_v1.1\glove.6B.100d.txt'  #path to glove word embeddings 
        #"/path/to img feat npz file"
        self.imgfeat=r'E:\Memex_QA\memexqa_dataset_v1.1\photos_inception_resnet_v2_l2norm.npz'
        self.outpath=r'E:\Memex_QA\prepro'   #"output path"
        self.testids=r'E:\Memex_QA\memexqa_dataset_v1.1\test_question.ids'    # "path to test id list"
        self.use_BERT=True
        self.valids = None    #"path to validation id list, if not set will be random 20%% of the training set"
        self.word_based = False   #"Word-based Embedding"
		

def get_args():
	# remember to put ur own paths here 
	parser = argparse.ArgumentParser(description="giving the original memoryqa dataset, will generate a *_data.p, *_shared.p for each dataset.")
	parser.add_argument("datajson",type=str,help="path to the qas.json")
	parser.add_argument("albumjson",type=str,help="path to album_info.json")
	parser.add_argument("testids",type=str,help="path to test id list")
	parser.add_argument("--valids",type=str,default=None,help="path to validation id list, if not set will be random 20%% of the training set")
	parser.add_argument("imgfeat",action="store",type=str,help="/path/to img feat npz file")
	parser.add_argument("outpath",type=str,help="output path")
	parser.add_argument("--word_based",action="store_true",default=False,help="Word-based Embedding")
	return parser.parse_args()


longformer = False

# if longformer :
# 	# Load the Longformer tokenizer and model
# 	tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# 	model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
# else:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# for each token with "-" or others, remove it and split the token 
# def process_tokens(tokens):
# 	newtokens = []
# 	l = ("-","/", "~", '"', "'", ":","\)","\(","\[","\]","\{","\}")
# 	for token in tokens:
# 		# split then add multiple to new tokens
# 		newtokens.extend([one for one in re.split("[%s]"%("").join(l),token) if one != ""])
# 	return newtokens

def l2norm(feat):
	l2norm = np.linalg.norm(feat,2)
	return feat/l2norm

# for Glove Word Embeddings preparation 
# word_counter words are lowered already
def get_word2vec(args,word_counter):
	word2vec_dict = {}
	import io
	with io.open(args.glove, 'r', encoding='utf-8') as fh:
		for line in fh:
			array = line.lstrip().rstrip().split(" ")
			word = array[0]
			vector = list(map(float, array[1:]))
			if word in word_counter:
				word2vec_dict[word] = vector
			#elif word.capitalize() in word_counter:
			#	word2vec_dict[word.capitalize()] = vector
			elif word.lower() in word_counter:
				word2vec_dict[word.lower()] = vector
			#elif word.upper() in word_counter:
			#	word2vec_dict[word.upper()] = vector

	#print "{}/{} of word vocab have corresponding vectors ".format(len(word2vec_dict), len(word_counter))
	return word2vec_dict
	
# this is for creating dirs quickly
def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

# HTML 
"""removes HTML tags from a given HTML string. 
   It achieves this by leveraging the HTMLParser 
   class from the html.parser module."""

class MLStripper(HTMLParser):
	def __init__(self):
		super().__init__()
		self.reset()
		self.fed = []
	
	def handle_data(self, d):
		"""method is overridden to capture text content within HTML tags."""
		"""Whenever the parser encounters text, it appends it to the fed list."""
		self.fed.append(d)
	
	"""get_data method joins all the captured text 
	   from the fed list into a single string and returns it."""
	def get_data(self):
		return ''.join(self.fed)
		

def strip_tags(html):
	"""an instance of the MLStripper class."""
	s = MLStripper()
	s.feed(html)        # first  fun
	return s.get_data() # second fun , returns the stripped text


# word_counter words are lowered already
def sentence2vec(sentence, word_based = False):
	if word_based:
		try:
			# using bert for tokenization and embedding 
			# inputs = tokenizer(sentence, return_tensors="pt")
			inputs = tokenizer(sentence, max_length=512, truncation=True, padding=True, return_tensors="pt")
			# print(inputs)

			"""The last hidden state of BERT is extracted for each token."""
			# feed forward path to the bert model
			outputs = model(**inputs)['last_hidden_state']
			# print(outputs)
			# print(outputs.view(-1 , 768))


			return outputs.view(-1, 768)
		except:
			return torch.zeros((1, 768))
	else:

		"""sentence based embedding"""
		try:
			# inputs = tokenizer(sentence, return_tensors="pt")
			inputs = tokenizer(sentence, max_length=512, truncation=True, padding=True , return_tensors="pt")
			outputs = model(**inputs)['pooler_output'] 

			# or we can pool it automatically 
			# Mean pooling
			# sentence_embedding = torch.mean(outputs.last_hidden_state , dim=1)
			# return sentence_embedding
			
			return outputs.view(-1)
		
		except:
			return torch.zeros(768)


# def sentence2vec(sentence, word_based):
    # if word_based:
    #     try:
    #         # Tokenize and encode the sentence, truncating if necessary
    #         inputs = tokenizer(sentence, return_tensors="pt", max_length=4096, truncation=True, padding=True)
            
    #         # Extract the last hidden state of the Longformer model
    #         outputs = model(**inputs)['last_hidden_state']
            
    #         # Reshape to [num_tokens, hidden_size]
    #         return outputs.view(-1, model.config.hidden_size)
    #     except Exception as e:
    #         print(f"Error during word-based embedding extraction: {e}")
    #         return torch.zeros((1, model.config.hidden_size))
    # else:
    #     try:
    #         # Sentence-based embedding
    #         inputs = tokenizer(sentence, return_tensors="pt", max_length=4096, truncation=True, padding=True)
    #         outputs = model(**inputs)

    #         # Use mean pooling for sentence representation
    #         sentence_embedding = torch.mean(outputs['last_hidden_state'], dim=1)
            
    #         return sentence_embedding.view(-1)
    #     except Exception as e:
    #         print(f"Error during sentence-based embedding extraction: {e}")
    #         return torch.zeros(model.config.hidden_size)
		


# prepro_each(args,"train",trainIds)
def prepro_each(args, data_type, question_ids):
	"""this function is 2 parts , the first is for dealing with the 
		training examples(Questions-Ans)
		the second part is for dealing the Albums itself which is accessed by
		the Questions ...
	"""    
	# mapping to make the question_id is the key   # qas.json 
	qas = {str(qa['question_id']): qa for qa in args.qas}    
	# for statistical purposes 
	global_aids = {} # all the album Id the question used, also how many question used that album
	questions,  questions_embed,  answers = [], [], []
	aid, qid, choices = [], [], []
	yidx = []
	"""for ex : trainIds [100293 , 390420 , 94303 , .... ] """
	for idx, question_id in enumerate(tqdm(question_ids)):
		qa = qas[question_id] #get the example
		# question
		# qa['question'] = 'Where did we travel to ?'
		# qi = ['Where', 'did', 'we', 'travel', 'to', '?']
		# cqi = [['W', 'h', 'e', 'r', 'e'], ['d', 'i', 'd'], ['w', 'e'], ['t', 'r', 'a', 'v', 'e', 'l'], ['t', 'o'], ['?']]
		question = qa['question']
		question_embed = sentence2vec(qa['question'],
								      word_based = True)

		# album ids
		for albumId in qa['album_ids']:
			albumId = str(albumId)
			if albumId not in global_aids:
				"""as it is dict"""
				"""initialization for the first time"""
				
				global_aids[albumId] = 0
			global_aids[albumId]+=1 # remember how many times this album is used

		# answer, choices
		# qa['answer'] = 'waco'
		# yi = ['Waco']
		# cyi = [['W', 'a', 'c', 'o']]
		answer = sentence2vec(qa['answer'], word_based = True)

		# ci = [['Uis', 'homecoming'], ['Tahoe'], ['Folly', 'beach']]
		# cci = [[['U', 'i', 's'], ['h', 'o', 'm', 'e', 'c', 'o', 'm', 'i', 'n', 'g']], [['T', 'a', 'h', 'o', 'e']], [['F', 'o', 'l', 'l', 'y'], ['b', 'e', 'a', 'c', 'h']]]
		choice = qa['multiple_choices_4'][:] # copy it
		# remove the answer in choices
		yidxi = choice.index(qa['answer']) # this is for during testing, we need to reconstruct the answer in the original order
		choice.remove(qa['answer']) # will error if answer not in choice
		choice = [sentence2vec(c, word_based = True) for c in choice]

		# Append
		questions.append(question)
		questions_embed.append(question_embed)
		answers.append(answer)
		aid.append([str(album_id) for album_id in qa['album_ids']])
		qid.append(question_id)
		choices.append(choice)
		yidx.append(yidxi)
     

    # make the album_id is the master of the album
	albums = {str(album['album_id']):album for album in args.albums}	
	album_info, pid2feat = {}, {}
    # keys which are the albums id , these are the used albums
	for albumId in tqdm(global_aids):
		album = albums[albumId]
		# creating new dictionary to contain the embedding of the album content
		temp = {'aid':album['album_id']}
		# album info
		temp['title'] = sentence2vec(album['album_title'], args.word_based)
		temp['description'] = sentence2vec(strip_tags(album['album_description']), args.word_based)
		# use _ to connect?
		# TherE is Some MEssing WhEreS
		if album['album_where'] is None:
			temp['where'] = torch.zeros(768)
		else:
			temp['where'] = sentence2vec(album['album_where'], args.word_based)
		
		temp['when'] = sentence2vec(album['album_when'], args.word_based)

		# photo info
		temp['photo_urls'] = [url for url in album['photo_urls']]
		temp['photo_titles'] = [sentence2vec(title, args.word_based) for title in album['photo_titles']]
		temp['photo_ids'] = [str(pid) for pid in album['photo_ids']]


		for pid in temp['photo_ids']:
			if pid not in pid2feat:
				pid2feat[pid] = args.images[pid]

		# this is a new one-one mapping {'album_id' : it's embedded content}
		album_info[albumId] = temp

	# from QA examples
	data = {
		'questions':questions,
		'questions_embed':questions_embed,
		'answers':answers,
		'yidx': yidx,
		'aid':aid,
		'qid':qid,
		'choices':choices
	}

    # album info which will be accessed by the QA
	shared = {
		"albums" :album_info, # albumId -> photo_ids/title/when/where ...
		"pid2feat":pid2feat, # pid -> image feature
	}

                #   "train"
	print(f"data:{data_type},album: {len(album_info)}/{len(albums)}, image_feat:{len(pid2feat)}")

	with open(os.path.join(args.outpath, f"{data_type}_data.p"), "wb") as fp:
		pickle.dump(data, fp)

	with open(os.path.join(args.outpath, f"{data_type}_shared.p"), "wb") as fp:
		pickle.dump(shared, fp)


def getTrainValIds(qas, validlist, testidlist):
	testIds = [one.strip() for one in open(testidlist,"r").readlines()]

	valIds = []
	if validlist is not None:
		valIds = [one.strip() for one in open(validlist,"r").readlines()]

	trainIds = []
	for one in qas:
		qid = str(one['question_id'])
		if((qid not in testIds) and (qid not in valIds)):
			trainIds.append(qid)

    # let's make validation set
	if validlist is None:
		valcount = int(len(trainIds)*0.2)
		random.seed(1)
		random.shuffle(trainIds)
		random.shuffle(trainIds)
		# split train to train and val 
		valIds = trainIds[:valcount]
		trainIds = trainIds[valcount:]
		
	print(f"total trainId : {len(trainIds)} valId : {len(valIds)} , testId : {len(testIds)} , total qa : {len(qas)}")
	return trainIds,valIds,testIds


def pickle_some():
	ids = ["170014" , "170003" , "170020" , "170026" , "170031" ,
		   "170000" , "170001" , "170002" , "170004" , "170005" ]
	return ids 


class to_pass:
	def __init__(self):
		self.qas = None
		self.albums = None
		self.images = None
		self.outpath=r'E:\Memex_QA\prepro'   #"output path"
		self.word_based = False   #"Word-based Embedding"




def main1():
	# use_BERT=True
	args = get_args()
	mkdir(args.outpath)
	args.qas = json.load(open(args.datajson,"r"))
	args.albums = json.load(open(args.albumjson,"r"))
	# if the image is a .p file, then we will read it differently
	# Length = 5090
	# map -> id : feature
	# 5739189334 : shape=2537
	if(args.imgfeat.endswith(".p")):
		print("read pickle image feat.")
		imagedata = pickle.load(open(args.imgfeat,"r"))
		args.images = {}
		assert len(imagedata[0]) == len(imagedata[1])
		for i,pid in enumerate(imagedata[0]):
			args.images[pid] = imagedata[1][i]
	else:
		print("read npz image feat.")
		args.images = np.load(args.imgfeat)
	# trainIds = 80% training data
	# valIds   = 20% training data
	# testIds  = args.testids = memexqa_dataset_v1.1/test_question.ids
	trainIds,valIds,testIds = getTrainValIds(args.qas,args.valids,args.testids)

	prepro_each(args,"train",trainIds)
	prepro_each(args,"val",valIds)
	prepro_each(args,"test",testIds)



def main2():
	# albumjson= r'E:\Memex_QA\memexqa_dataset_v1.1\album_info.json'
	# datajson=r'E:\Memex_QA\memexqa_dataset_v1.1\qas.json'
	# glove='memexqa_dataset_v1.1/glove.6B.100d.txt' 
	# imgfeat=r'E:\Memex_QA\memexqa_dataset_v1.1\photos_inception_resnet_v2_l2norm.npz'
	# outpath=r'E:\Memex_QA\prepro'
	# testids=r'E:\Memex_QA\memexqa_dataset_v1.1\test_question.ids'
	# use_BERT=True
	# valids = None
	# word_based = False

	cfg = config()
	tp = to_pass()
	# mkdir(args.outpath)
	
	# Length = 20563
	# 'evidence_photo_ids': ['1164554'], 
	# 'question': 'What did we do?', 
	# 'secrete_type': 'single_album', 
	# 'multiple_choices_20': ['February 26 2005', 'May 28 2006', 'December 18 2004', 'October 29 2004', 'In bushes', 'Funeral', 'Springer wedding', 'Jacksonville fl', '1 of the kids', 'Georgia may jagger', 'Lauren', 'Rhubarb', 'Watch red sox parade', 'White', "Celebrating liza's birthday", 'Blue stuffed animal', '14', '3', 'Twice', '4 tiers'], 
	# 'multiple_choices_4': ['Watch red sox parade', 'White', "Celebrating liza's birthday", 'Blue stuffed animal'], 
	# 'secrete_hit_id': '3W1K7D6QSB4YZRFASL9VJQP73FUBZL', 
	# 'secrete_aid': '29851', 
	# 'album_ids': ['29851'], 
	# 'answer': 'Watch red sox parade', 
	# 'question_id': 170000, 
	# 'flickr_user_id': '35034354137@N01'
	
	"""let's override the path cfg.qas"""
	tp.qas = json.load(open(cfg.qas,"r"))
    
	# qas = json.load(open(datajson,"r")) # before making args 
	# print(len(qas))   20563
	# print(type(qas))  list
	# print(qas[0]) # show the first ex
	"""
		{'evidence_photo_ids': ['1164554'],
		'question': 'What did we do?',
		'secrete_type': 'single_album', 
		'multiple_choices_20': ['February 26 2005', 'May 28 2006', 'December 18 2004', 'O_album', 'multiple_choices_20': ['February 26 2005', 'May 28 2006', 'December 18 2004', 'October 29 2004', 'In bushes', 'Funeral', 'Springer wedding', 'Jacksonville fl', '1 of the kids', 'Georgia may jagger', 'Lauren', 'Rhubarb', 'Watch red sox parade', 'White', "Celebrating liza's birthday", 'Blue stuffed animal', '14', '3', 'Twice', '4 tiers'],
		'multiple_choices_4': ['Watch red sox parade', 'White', "Celebrating liza's birthday", 'Blue stuffed animal'],
		'secrete_hit_id': '3W1K7D6QSB4YZRFASL9VJQP73FUBZL', 
		'secrete_aid': '29851', 
		'album_ids': ['29851'],
		'answer': 'Watch red sox parade', 
		'question_id': 170000, 
		'flickr_user_id': '35034354137@N01'}
	"""

	# Length = 630
	# 'album_where': 'New York, 10009, USA', 
	# 'photo_urls': ['https://farm3.staticflickr.com/2762/4513010720_0f5aacacbf_o.jpg', 'https://farm3.staticflickr.com/2127/4513022954_64334f780e_o.jpg', 'https://farm3.staticflickr.com/2024/4513024686_19204de865_o.jpg', 'https://farm3.staticflickr.com/2743/4512372271_bb7b188d47_o.jpg', 'https://farm3.staticflickr.com/2286/4512373755_d71b7d82c9_o.jpg', 'https://farm3.staticflickr.com/2739/4512366991_3ca24f3105_o.jpg'], 
	# 'album_id': '72157623710621031', 
	# 'photo_gps': [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 
	# 'photo_titles': ['Blowing Out Candles', 'Eden and Lulu', 'Party Stump', 'Paper Crown I', 'Craft Table', 'Slice of Cake'], 
	# 'album_description': 'April 11, 2010, at La Plaza Cultural community garden.  Lulu was in the 2s class at Little Missionary when Eden was in the 4s, and they just took a shine to each other.', 
	# 'album_when': 'on April 11 2010', 
	# 'flickr_user_id': '10485077@N06', 
	# 'album_title': "Lulu's 4th Birthday", 
	# 'photo_tags': ['birthday nyc newyorkcity eastvillage ny newyork downtown lulu manhattan birthdayparty jc alphabetcity somethingsweet saoirse laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork downtown lulu manhattan birthdayparty eden alphabetcity somethingsweet laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork cake downtown lulu manhattan birthdayparty birthdaycake stump eden alphabetcity somethingsweet laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork downtown princess manhattan birthdayparty crown eden alphabetcity laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork painting downtown lulu manhattan crafts birthdayparty eden alphabetcity crowns laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork cake downtown manhattan birthdayparty birthdaycake eden alphabetcity somethingsweet sliceofcake laplazacultural'], 
	# 'photo_captions': ["at [female] 's birthday party , they ate cake .", 'they had ice cream too .', 'then they got to do some crafts .', '[female] made a silly hat .', 'everybody had fun making things .', 'she made fun art projects .'], 
	# 'photo_ids': ['4513010720', '4513022954', '4513024686', '4512372271', '4512373755', '4512366991']}
	
	tp.albums = json.load(open(cfg.albumjson,"r"))

	# albums = json.load(open(albumjson,"r")) # before making args 
	# print(len(albums)) 630
	# print(type(albums)) list
	# print(albums[0])
	# print(albums[0]['album_where']) New York, 10009, USA

	"""

	{'album_where': 'New York, 10009, USA',
		'photo_urls': ['https://farm3.staticflickr.com/2762/4513010720_0f5aacacbf_o.jpg', 'https://farm3.staticflickr.com/2127/4513022954_64334f780e_o.jpg', 'https://farm3.staticflickr.com/2024/4513024686_19204de865_o.jpg', 'https://farm3.staticflickr.com/2743/4512372271_bb7b188d47_o.jpg', 'https://farm3.staticflickr.com/2286/4512373755_d71b7d82c9_o.jpg', 'https://farm3.staticflickr.com/2739/4512366991_3ca24f3105_o.jpg'],
		'album_id': '72157623710621031', 
		'photo_gps': [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
		'photo_titles': ['Blowing Out Candles', 'Eden and Lulu', 'Party Stump', 'Paper Crown I', 'Craft Table', 'Slice of Cake'], 
		'album_description': 'April 11, 2010, at La Plaza Cultural community garden.  Lulu was in the 2s class at Little Missionary when Eden was in the 4s, and they just took a shine to each other.',
		'album_when': 'on April 11 2010',
		'flickr_user_id': '10485077@N06',
		'album_title': "Lulu's 4th Birthday",
		'photo_tags': ['birthday nyc newyorkcity eastvillage ny newyork downtown lulu manhattan birthdayparty jc alphabetcity somethingsweet saoirse laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork downtown lulu manhattan birthdayparty eden alphabetcity somethingsweet laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork cake downtown lulu manhattan birthdayparty birthdaycake stump eden alphabetcity somethingsweet laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork downtown princess manhattan birthdayparty crown eden alphabetcity laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork painting downtown lulu manhattan crafts birthdayparty eden alphabetcity crowns laplazacultural', 'birthday nyc newyorkcity eastvillage ny newyork cake downtown manhattan birthdayparty birthdaycake eden alphabetcity somethingsweet sliceofcake laplazacultural'], 'photo_captions': ["at [female] 's birthday party , they ate cake .", 'they had ice cream too .', 'then they got to do some crafts .', '[female] made a silly hat .', 'everybody had fun making things .', 'she made fun art projects .'], 
		'photo_ids': ['4513010720', '4513022954', '4513024686', '4512372271', '4512373755', '4512366991']}

	"""

	# if the image is a .p file, then we will read it differently
	# Length = 5090
	# map -> id : feature
	# 5739189334 : shape=2537
	# if(args.imgfeat.endswith(".p")):
	# 	print("read pickle image feat.")
	# 	imagedata = pickle.load(open(cfg.imgfeat,"r"))
	# 	cfg.images = {}
	# 	assert len(imagedata[0]) == len(imagedata[1])
	# 	for i,pid in enumerate(imagedata[0]):
	# 		 images[pid] = imagedata[1][i]
	# else:
	

	# .npz
	print("read npz image feat.")
	tp.images = np.load(cfg.imgfeat)


	# print(len(images))5090
	# List keys
	# print("Keys:", images.files)
    
	"""here's how if u want to print specific feature vec"""
	# keys = images.files 
	# idx = keys[0] 
	# print(len(images[idx]))  # 2537

	# Iterate over and print all arrays
	# for key in images.files:
	#     print(f"{key}: {images[key]}")



	# trainIds = 80% training data
	# valIds   = 20% training data
	# testIds  = args.testids = memexqa_dataset_v1.1/test_question.ids
	
	# trainIds , valIds , testIds = getTrainValIds(args.qas ,args.valids , args.testids)

	# prepro_each(args,"train",trainIds)
	# prepro_each(args,"val",valIds)
	# prepro_each(args,"test",testIds)


    # for testing the model before training
	ids_to_pickle = pickle_some()
	prepro_each(tp , "Unified_multi_albums" , ids_to_pickle)


if __name__ == "__main__":

	# """this config for usage if u will not use argument parsers"""
	use_parsers = False
	if use_parsers : main1()
	else : main2()
