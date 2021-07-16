#!/usr/bin/python
# -*- coding: utf-8 -*
import tensorflow as tf
import os

from tensorflow import keras 
from flask import request
from flask import render_template
from flask import Flask, request, Response
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

DEVICE = "cuda"
num2word = None
word2num = None
MODEL = None
class Diction(): 
	def __init__(self):
		self.path ='./簡媜'
		self.data = []
		self.words = ""
	def read(self):
		for i in range(1, 41):
			with open(self.path+'/'+str(i)+".txt.txt") as f:
				lines = f.readlines()
				count = 0
				while count < len(lines):
					if lines[count] == '\n':
						del lines[count]
					count += 1
				self.data += lines
		for line in self.data:
			tmp = line.replace('\n', '').replace('-', '').replace(' ', '')
			self.words += tmp
		n = len(self.words)
		w = len(set(self.words))
		print(f"簡媜散文共有 {n} 中文字")           
		print(f"包含了 {w} 個獨一無二的字")
	def create_diction(self):
		self.num_words = len(set(self.words))
		self.tokenizer=tf.keras.preprocessing.text .Tokenizer( num_words=self.num_words, char_level=True,filters='')
		self.tokenizer.fit_on_texts(self.words)
		word_to_num = {}
		for key, value in  self.tokenizer.index_word.items():
			word_to_num[value] = key
		num_to_word = {}
		for key, value in self.tokenizer.index_word.items():
			num_to_word[key] = value
		return num_to_word, word_to_num
                
                
                
class Model():
	def __init__(self):
		self.path = './first.h5'
		self.EMBEDDING_DIM = 512
		self.RNN_UNITS = 1024
		self.BATCH_SIZE = 1
		self.num_words = 3111
		self.infer_model =None
	def build(self):
		
		self.infer_model = tf.keras.Sequential()
		self.infer_model.add(
		    tf.keras.layers.Embedding(
		        input_dim=self.num_words, 
		        output_dim=self.EMBEDDING_DIM,
		        batch_input_shape=[
		            self.BATCH_SIZE, None]
		))
		
		# LSTM 層
		self.infer_model.add(
		    tf.keras.layers.LSTM(
		    units=self.RNN_UNITS, 
		    return_sequences=True, 
		    stateful=True
		))
		
		# 全連接層
		self.infer_model.add(
		    tf.keras.layers.Dense(
		        self.num_words))
		
		# 讀入之前訓練時儲存下來的權重
		self.infer_model.load_weights(self.path)
		self.infer_model.build(
		    tf.TensorShape([1, None]))
		    
	def predict(self, string, output_length, temperature):
		count = 0
		ans =string
		s = word2num[string[-1]]
		while count < output_length:
			seed_indices = [s] 
			Input = tf.expand_dims( seed_indices, axis=0)

			predictions = self.infer_model(Input)
			predictions = tf.squeeze(predictions, 0)
			predictions /= temperature
			sampled_indices = tf.random.categorical( predictions, num_samples=1)
			s = np.array(sampled_indices)[0][0]
			print(s)
			output =num2word[s]
			print(output, end = '')
			ans +=output
			count +=1
			if count%50==0:
				print()
		return ans
				

@app.route("/", methods=["GET", "POST"])
def upload_predictions():
	if request.method == "POST":
		string = request.form['string']
		length = request.form['length']
		temperature = request.form['temperature']
		print(type(string),type(length), type(temperature))
		try:
			a = word2num[string[-1]]
		except:
			return "the word doesn't match,try another word", 404
		try:
			a = int(length)
			b = float(temperature)
		except:
			return "length or temperature must be numebr.", 404
			
		if int(length) <=0 or int(length) >2000:
			return "the length must between  1 and 2000",404
		if 0 > float(temperature) or float(temperature)>1 :
			return "the temperature must between  0 and 1", 404
		output = MODEL.predict(str(string),int(length),float(temperature) ) 
		return render_template("generator.html", output=output)
	return render_template("generator.html",output=None)
				
if __name__ == '__main__':
	diction = Diction()
	diction.read()
	num2word , word2num = diction.create_diction()
	print(type(num2word), type(word2num))
	test = Model()
	test.build()
	MODEL = test
	# test.predict(1, 20,0.1)
	app.run(port=7070,debug=True)

	
			
