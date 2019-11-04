import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import tflearn
import random
import json

with open("intents.json") as file:
	data = json.load(file)

for i in data["intents"][0]:
	print(i," ",data["intents"][0][i])

def write_in_file(command, correct_response, category):
	for i in data["intents"]:
		if i["tag"]==category:
			i["patterns"].append(command)
			i["responses"].append(correct_response)
			with open("intents.json","w") as outfile:
				json.dump(data, outfile)
			return
	command_list = []
	response_list = []
	command_list.append(command)
	response_list.append(correct_response)
	data["intents"].append({
		"tag":category,
		"patterns":command_list,
		"responses":response_list,
		"context_set":""
		})
	with open("intents.json","w") as outfile:
	 	json.dump(data, outfile)

class dataset:
	def __init__(self):
		print("constructor called, new dataset created.")
		self.words = []
		self.labels = []
		self.docs_x = []
		self.docs_y = []
		self.training = []
		self.output = []

	def display_data(self):
		print("words")
		print(self.words)
		print("labels")
		print(self.labels)
		print("docs_x")
		print(self.docs_x)
		print("docs_y")
		print(self.docs_y)
		print("training")
		print(self.training)
		print("output")
		print(self.output)

	def clean_object(self):
		self.words = []
		self.labels = []
		self.docs_x = []
		self.docs_y = []
		self.training = []
		self.output = []
  
class tensorflow_model:
	def __init__(self,dataset_object):
		tf.reset_default_graph()
		net = tflearn.input_data(shape=[None, len(dataset_object.training[0])])
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, len(dataset_object.output[0]),activation="softmax")
		net = tflearn.regression(net)
		self.model = tflearn.DNN(net)

	def modify_model(self,dataset_object):
		tf.reset_default_graph()
		net = tflearn.input_data(shape=[None, len(dataset_object.training[0])])
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, len(dataset_object.output[0]),activation="softmax")
		net = tflearn.regression(net)
		self.model = tflearn.DNN(net)

def processing(dataset_object):

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			dataset_object.words.extend(wrds)
			dataset_object.docs_x.append(wrds)
			dataset_object.docs_y.append(intent["tag"])

		if intent["tag"] not in dataset_object.labels:
			dataset_object.labels.append(intent["tag"])

	dataset_object.words = [stemmer.stem(w.lower()) for w in dataset_object.words]
	dataset_object.words = sorted(list(set(dataset_object.words)))
	dataset_object.labels = sorted(dataset_object.labels)

	out_empty = [0 for _ in range(len(dataset_object.labels))]
	for x, doc in enumerate(dataset_object.docs_x):
		print(x,doc)
		bag = []
		wrds = [stemmer.stem(w) for w in doc if w != '?']
		print(wrds)
		for w in dataset_object.words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)
		print(len(bag))
		output_row = out_empty[:]
		output_row[dataset_object.labels.index(dataset_object.docs_y[x])] = 1
		dataset_object.training.append(bag)
		dataset_object.output.append(output_row)
	training = np.array(dataset_object.training)
	output = np.array(dataset_object.output)



def training(dataset_object, model_object):
	model_object.model.fit(dataset_object.training,dataset_object.output,
		n_epoch=1000, batch_size=8, show_metric=True)
	model_object.model.save("model.tflearn")

def testing(dataset_object, model_object, command):
	user_words = nltk.word_tokenize(command)
	user_input = []
	bag = []
	user_words = [stemmer.stem(w.lower()) for w in user_words if w != '?']
	print(user_words)
	for w in dataset_object.words:
		if w in user_words:
			bag.append(1)
		else:
			bag.append(0)
		
	user_input.append(bag)
	user_input = np.array(user_input)

	result = model_object.model.predict(user_input)
	result = result.tolist()
	result_index = result[0].index(max(result[0]))
	query_type = dataset_object.labels[result_index]
	for intent in data["intents"]:
		if intent["tag"]==query_type:
			print(random.choice(intent["responses"]))
			#for response in intent["responses"]:
			#	print(response)

	print("Am I wrong")
	while(1):
		ans = input()
		if(ans.lower()=="yes"):
			print("kindly write the right response. I'll learn it")
			correct_response = input()
			print("What category should I put it into?")
			print("I have following categories of queries")
			for intent in data["intents"]:
				print(intent["tag"])
			print('choose one or add new')
			category = input()	
			write_in_file(command,correct_response,category)
			print("training again")
			dataset_object.clean_object()
			processing(dataset_object)
			model_object.modify_model(dataset_object)
			training(dataset_object, model_object)
			break
		elif ans.lower()=="no":
			print("Thank you keep communicating with me")
			break
		else:
			print("sorry I didn't understand")

## User input
if __name__=="__main__":
	dataset_object = dataset()
	processing(dataset_object)
	model_object = tensorflow_model(dataset_object)
	training(dataset_object, model_object)
	command = input()
	while command != "exit":
		testing(dataset_object, model_object, command)
		command = input()























