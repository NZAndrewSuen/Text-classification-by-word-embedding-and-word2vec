import string
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
 
# load doc into memory
def load_doc(filename):
	file = open(filename, 'r', encoding='utf8')
	text = file.read()
	file.close()
	return text
 
# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	doc = load_doc(filename)
	tokens = clean_doc(doc)
	vocab.update(tokens)
 
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		path = directory + '/' + filename
		add_doc_to_vocab(path, vocab)

# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

#define vocab
vocab = Counter()
#add all docs to vocab
process_docs('imdb/neg',vocab,True)
process_docs('imdb/poz',vocab,True)
#print the size of the vocab
print(len(vocab))

# keep tokens with a min occurrence
min_occurane = 3
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')