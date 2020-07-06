from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
plt.style.use('ggplot')
 
# load doc into memory
def load_doc(filename):
	file = open(filename, 'r', encoding = 'unicode_escape')
	text = file.read()
	file.close()
	return text
 
# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation   from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens
	
 
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('9'):
			continue
		if not is_trian and not filename.startswith('9'):
			continue
		path = directory + '/' + filename
		doc = load_doc(path)
		tokens = clean_doc(doc, vocab)
		documents.append(tokens)
	return documents

def plot_history(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	x = range(1, len(acc) + 1)

	plt.figure(figsize=(12,5))
	plt.subplot(1,2,1)
	plt.plot(x, acc, 'b', label = 'Training acc')
	plt.plot(x, val_acc, 'r', label='Test acc')
	plt.title('Training and Test acc')
	plt.legend()
	plt.subplot(1,2,2)
	plt.plot(x, loss, 'b', label='Training loss')
	plt.plot(x, val_loss, 'r', label='Test loss')
	plt.title('Training and Test loss')
	plt.legend()
	plt.show()

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
 
# load all training reviews
positive_docs = process_docs('imdb/poz', vocab, True)
negative_docs = process_docs('imdb/neg', vocab, True)
train_docs = negative_docs + positive_docs
 
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)
 
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
 
# load all test reviews
positive_docs = process_docs('imdb/poz', vocab, False)
negative_docs = process_docs('imdb/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
 
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
 
# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=3, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtrain, ytrain, verbose=0)
print('Training Accuracy: %f' % (acc*100))
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Testing Accuracy: %f' % (acc*100))
plot_history(history)
