"""
You can ignore this script if you will load the data into the TrainDataset 
directlty here : src\dataset\train_dataset.py
"""
import re
import os
import sys
import json
import nltk
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

# import cPickle as pickle
import pickle

import argparse
from html.parser import HTMLParser

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# .p is refering for pickle extension

def get_args():
	# remember to put ur own paths here 
	parser = argparse.ArgumentParser(description="giving the original memoryqa dataset, will generate a *_data.p, *_shared.p for each dataset.")
	parser.add_argument("datajson",type=str,help="path to the qas.json")
	parser.add_argument("albumjson",type=str,help="path to album_info.json")
	parser.add_argument("testids",type=str,help="path to test id list")
	parser.add_argument("--valids",type=str,default=None,help="path to validation id list, if not set will be random 20%% of the training set")
	parser.add_argument("imgfeat",action="store",type=str,help="/path/to img feat npz file")
	parser.add_argument("outpath",type=str,default = r"E:\Memex_QA" , help="output path")
	parser.add_argument("--word_based",action="store_true",default=False,help="Word-based Embedding")
	return parser.parse_args()



# this is for creating dirs quickly
def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


# return the L2 norm for feature vector
def l2norm(feat):
	l2norm = np.linalg.norm(feat,2)
	return feat/l2norm


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
def sentence2vec(sentence, word_based):
	if word_based:
		try:
			# using bert for tokenization and embedding 
			inputs = tokenizer(sentence, return_tensors="pt")

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
			inputs = tokenizer(sentence, return_tensors="pt")			
			outputs = model(**inputs)['pooler_output'] 

			# or we can pool it automatically 
			# Mean pooling
			# sentence_embedding = torch.mean(outputs.last_hidden_state , dim=1)
			# return sentence_embedding
			
			return outputs.view(-1)
		
		except:
			return torch.zeros(768)



# prepro_each(args,"train",trainIds)
def prepro_each(qass , data_type, question_ids, word_based , images , outpath, albumss):
	"""this function is 2 parts , the first is for dealing with the 
		training examples(Questions-Ans)
		the second part is for dealing the Albums itself which is accessed by
		the Questions ...
	""" 
		  
     
	# mapping to make the question_id is the key   # qas.json 
	qas = {str(qa['question_id']): qa for qa in qass}
     
	# for statistical purposes, to know which albums used how many times 
	global_aids = {} # all the album Id the question used, also how many question used that album

	questions,  questions_embed,  answers = [], [], []
	# album_id,  question_id,   4-choices
	aid, qid, choices = [], [], []
	# answer index of the 4 indeces
	yidx = []
        

	"""for ex : trainIds [100293 , 390420 , 94303 , .... ] """
	for idx, question_id in enumerate(tqdm(question_ids)):
		qa = qas[question_id] #get the example

		# question
		# qa['question'] = 'Where did we travel to ?'
		# qi = ['Where', 'did', 'we', 'travel', 'to', '?']
		# this is for char embedding purposes which they did in the FVTA paper
		# cqi = [['W', 'h', 'e', 'r', 'e'], ['d', 'i', 'd'], ['w', 'e'], ['t', 'r', 'a', 'v', 'e', 'l'], ['t', 'o'], ['?']]
		question = qa['question']
		question_embed = sentence2vec(qa['question'],
								         word_based)



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
		answer = sentence2vec(qa['answer'], word_based)

		# ci = [['Uis', 'homecoming'], ['Tahoe'], ['Folly', 'beach']]
		# cci = [[['U', 'i', 's'], ['h', 'o', 'm', 'e', 'c', 'o', 'm', 'i', 'n', 'g']], [['T', 'a', 'h', 'o', 'e']], [['F', 'o', 'l', 'l', 'y'], ['b', 'e', 'a', 'c', 'h']]]
		choice = qa['multiple_choices_4'][:] # copy it
		# remove the answer in choices
		yidxi = choice.index(qa['answer']) # this is for during testing, we need to reconstruct the answer in the original order
		choice.remove(qa['answer']) # will error if answer not in choice
		choice = [sentence2vec(c,   word_based) for c in choice]

		# Append
		questions.append(question)
		questions_embed.append(question_embed)
		answers.append(answer)

		aid.append([str(album_id) for album_id in qa['album_ids']])
		qid.append(question_id)
		choices.append(choice)
		yidx.append(yidxi)
     


    # make the album_id is the master of the album
	albums = {str(album['album_id']):album  for album in albumss}
	
	album_info, pid2feat = {}, {}
	                    # keys which are the albums id , these are the used albums
	for albumId in tqdm(global_aids):
		album = albums[albumId]

		# creating new dictionary to contain the embedding of the album content
		temp = {'aid':album['album_id']}

		# album info
		temp['title'] = sentence2vec(album['album_title'],  word_based)
		temp['description'] = sentence2vec(strip_tags(album['album_description']), word_based)

		# use _ to connect?
		# TherE is Some MEssing WhEreS
		if album['album_where'] is None:
			temp['where'] = torch.zeros(768)
		else:
			temp['where'] = sentence2vec(album['album_where'], word_based)
		
		temp['when'] = sentence2vec(album['album_when'], word_based)

		# photo info
		temp['photo_urls'] = [url for url in album['photo_urls']]
		# temp['photo_urls'] = album['photo_urls']     ,but i preferred the above to show that there is a list 
		temp['photo_titles'] = [sentence2vec(title, word_based) for title in album['photo_titles']]
		temp['photo_ids'] = [str(pid) for pid in album['photo_ids']]


		for pid in temp['photo_ids']:
			# focus
			if pid not in pid2feat:
				pid2feat[pid] = images[pid]
		

		# this is a new one-one mapping {'album_id' : it's embedded content}
		album_info[albumId] = temp
 

    
	"""all the lists and dictionaries defined above let's combine them in one place"""
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

	with open(os.path.join(outpath, f"{data_type}_data.p"), "wb") as fp:
		pickle.dump(data, fp)

	with open(os.path.join(outpath, f"{data_type}_shared.p"), "wb") as fp:
		pickle.dump(shared, fp)




def getTrainValIds(qas, validlist, testidlist):
	testIds = [one.strip() for one in open(testidlist,"r").readlines()]

	valIds = []
	if validlist is not None:
		valIds = [one.strip() for one in open(validlist,"r").readlines()]


	trainIds = [] # this actually will contain train ids and val ids
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
	return trainIds, valIds, testIds 
    # total trainId : 13620 valId : 3404 , testId : 3539 , total qa : 20563


     


if __name__ == "__main__" :
    # args = get_args()


	albumjson= r'E:\Memex_QA\memexqa_dataset_v1.1\album_info.json'
	datajson=r'E:\Memex_QA\memexqa_dataset_v1.1\qas.json'
	glove='memexqa_dataset_v1.1/glove.6B.100d.txt' 
	imgfeat=r'E:\Memex_QA\memexqa_dataset_v1.1\photos_inception_resnet_v2_l2norm.npz'
	outpath=r'E:\Memex_QA\prepro'
	testids=r'E:\Memex_QA\memexqa_dataset_v1.1\test_question.ids'
	use_BERT=True
	valids = None
	word_based = True

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


	qas = json.load(open(datajson,"r"))
	# print(len(qas))   20563
	# print(type(qas))  list
	# print(qas[0])
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


	albums = json.load(open(albumjson,"r"))
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


	if(imgfeat.endswith(".p")):
		print("read pickle image feat.")
		imagedata = pickle.load(open(imgfeat,"r"))
		images = {}
		assert len(imagedata[0]) == len(imagedata[1])
		for i,pid in enumerate(imagedata[0]):
			images[pid] = imagedata[1][i]
	else:
		# .npz
		print("read npz image feat.")
		images = np.load(imgfeat)



	# print(len(images))5090
	# List keys
	# print("Keys:", images.files)
	keys = images.files
	idx = keys[0] 
	# print(len(images[idx]))  # 2537
	# Iterate over and print all arrays
	# for key in images.files:
	#     print(f"{key}: {images[key]}")





	# trainIds = 80% training data
	# valIds   = 20% training data
	# testIds  = args.testids = memexqa_dataset_v1.1/test_question.ids

	trainIds , valIds , testIds = getTrainValIds(qas , valids , testids)
	# print(trainIds[:10])
	# saving trainIds , valIds and testIds

	IDs={'trainIds' : trainIds,
		'valIds' : valIds ,
		'testIds' : testIds }

	# Save as JSON file
	# with open('IDs.json', 'w') as json_file:
	#      json.dump(IDs, json_file)
	json_file = open('IDs.json', 'w')
	json.dump(IDs, json_file)



	# prepro_each(qas , "train", trainIds, word_based , images , outpath, albums)
	# prepro_each(args,"val",valIds)
	# prepro_each(args,"test",testIds)