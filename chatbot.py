# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import re
import time

lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


############################ NLP #####################################
# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
	_line = line.split(' +++$+++ ')
	if len(_line) == 5:
		id2line[_line[0]] = _line[4]

# Creating a list of all conversations
conversations_ids = []

"""
Last row of the conversation dataset is an empty row. So take all rows
except the last one
"""
for conversation in conversations[:-1]:
	_conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
	"""
	u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']

	Only need the ['L194', 'L195', 'L196', 'L197'] => this portion from
	the conversations which is the last index. This is done in
	.split(' ++$++ ')[-1]

	['L194', 'L195', 'L196', 'L197'] => from here only needs the id,
	not [] or quotes. '[' is the 0th index. So will take from index 1
	upto everything. But won't take the ']' which is the last index.
	This is done in split(' +++$+++ ')[-1][1:-1] <= here

	need to replace quote and spaces
	"""

	conversations_ids.append(_conversation.split(','))


# Getting separately the questions and answers
questions = []
answers = []

for conversation in conversations_ids:
	for i in range(len(conversation) - 1):
		questions.append( id2line[conversation[i]] )
		# ['L194', 'L195', 'L196', 'L197'] L195 is the ans of L194
		answers.append( id2line[conversation[i+1]] )


# Clean text with regular expression
def clean_text(text):
	text = text.lower()
	text = re.sub(r"i'm", "i am", text)
	text = re.sub(r"he's", "he is", text)
	text = re.sub(r"she's", "she is", text)
	text = re.sub(r"that's", "that is", text)
	text = re.sub(r"what's", "what is", text)
	text = re.sub(r"where's", "where is", text)
	text = re.sub(r"there's", "there is", text)
	text = re.sub(r"'bout", "about", text)
	text = re.sub(r"it's", "it is", text)
	text = re.sub(r"doesn't", "does not", text)
	text = re.sub(r"don't", "do not", text)
	text = re.sub(r"didn't", "did not", text)
	text = re.sub(r"\'ll", " will", text)
	text = re.sub(r"\'ve", " have", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"\'d", " would", text)
	text = re.sub(r"won't", "will not", text)
	text = re.sub(r"can't", "cannot", text)
	text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)

	return text

# Clean questions
clean_questions = []

for question in questions:
	clean_questions.append( clean_text(question) )

# Clean answers
clean_answers = []

for ans in answers:
	clean_answers.append( clean_text(ans) )


# Creating a dict that maps each word to it's number of ocurrences
word2count = {}

for qus in clean_questions:
	for word in qus.split():
		if word not in word2count:
			word2count[word] = 1
		else:
			word2count[word] += 1

for ans in clean_answers:
	for word in ans.split():
		if word not in word2count:
			word2count[word] = 1
		else:
			word2count[word] += 1


# two dict to map the qus and ans words to unique integer
threshold = 20 # changable
questionswords2int = {}
word_number = 0 # 1st, 2nd, 3rd...

for word, count in word2count.items():
	if count >= threshold:
		questionswords2int[word] = word_number
		word_number += 1

answerswords2int = {}
word_number = 0 # 1st, 2nd, 3rd...

for word, count in word2count.items():
	if count >= threshold:
		answerswords2int[word] = word_number
		word_number += 1


# Last tokens to the dict
tokens = ['<PAD>', '<SOS>', '<EOS>', '<OUT>']

for token in tokens:
	questionswords2int[token]  = len(questionswords2int) + 1

for token in tokens:
	answerswords2int[token]  = len(answerswords2int) + 1


# inverse dict of answerswords2int
# for seq2seq model
# w_i is word count integer and w is word
# answerswords2int.items() returns in format key(w):val(w_in)
answersint2word = {w_i: w for w, w_i in answerswords2int.items()}


# add EOS at the end of every ans
for i in range(len(clean_answers)):
	clean_answers[i] += '<EOS>'


# Translating all the qus and ans to int
# and replacing all the words that're filtered out bu <OUT>

questions_into_int = []
for qus in clean_questions:
	ints = []
	for word in qus.split():
		if word not in questionswords2int:
			ints.append( questionswords2int['<OUT>'] )
		else:
			ints.append( questionswords2int[word] )
	questions_into_int.append( ints )

answers_into_int = []
for ans in clean_answers:
	ints = []
	for word in ans.split():
		if word not in answerswords2int:
			ints.append( answerswords2int['<OUT>'] )
		else:
			ints.append( answerswords2int[word] )
	answers_into_int.append( ints )


# Sorting questions and answers
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 26):
	for i in enumerate(questions_into_int):
		if len(i[1]) == length:
			sorted_clean_questions.append( questions_into_int[i[0]] )
			sorted_clean_answers.append( answers_into_int[i[0]] )

######################### NLP ######################################

######################### SEQ2SEQ MODEL ######################################

def model_inputs():
	inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
	targets = tf.placeholder(tf.int32, [None, None], name = 'target')
	lr = tf.placeholder(tf.float32, name = 'learning rate')

	# controls the drop out rate for neuron
	keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

	return inputs, targets, lr, keep_prob


def preprocess_targets(targets, word2int, batch_size):
	# every target row will have a unique token in the start which is
	# <SOS> which will be appended with the batch
	left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
	# won't take the last column as it is a unique term <EOS>
	right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
	# last param of concat() is axis. axis == 1 means horizontal concat
	# axis == 0 is vertical concat
	preprocessed_targets = tf.concat([left_side, right_side], 1)

	return preprocessed_targets

# RNN encoder layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
	lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
	lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
	encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
	_, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
													cell_bw = encoder_cell,
													sequence_length = sequence_length,
													inputs = rnn_inputs,
													dtype = tf.float32)

	return encoder_state






















