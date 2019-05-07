##Other imports
import json
import numpy as np

##Loading the intents file
with open('questions_tag.json') as json_file:
  intents = json.load(json_file)

##Importing NLTK essentials
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

##Organizing the documents
nltk.download('punkt')
words = []
classes = []
documents = []
ignore_words = ['?','@','$','&']
##Looping through each sentence
for intent in intents['intents']:
  for pat in intent['patterns']:
    
    w = nltk.word_tokenize(pat)
    words.extend(w)
    ## add to documents
    documents.append((w,intent["tag"]))
    ## addd to our class list
    if intent["tag"] not in classes:
      classes.append(intent["tag"])
                
## stem and lower each word
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

##Creating the training data
training = []
output = []
# empty array for output
output_empty = [0]*len(classes)
for doc in documents:
  bag = []
  pat_words = doc[0]
  pat_words = [stemmer.stem(word.lower()) for word in pat_words]
  for w in words:
    bag.append(1) if w in pat_words else bag.append(0)
    
  output_row = list(output_empty)
  output_row[classes.index(doc[1])] = 1
  training.append([bag,output_row])
  
  
# shuffle our features and turn into np.array
np.random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

##Building the model.
import tensorflow as tf
import keras

tf.reset_default_graph()

import tflearn
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1500, batch_size=8, show_metric=True)
model.save('model.tflearn')

# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('questions_tag.json') as json_data:
    intents = json.load(json_data)

def clean_up_sentence(sentence):
  
  # tokenize the pattern
  sentence_words = nltk.word_tokenize(sentence)
  # stem each word
  sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
  
  # tokenize the pattern
  sentence_words = clean_up_sentence(sentence)
  # bag of words
  bag = [0]*len(words)  
  for s in sentence_words:
    for i,w in enumerate(words):
      if w == s:
        
        bag[i] = 1
        if show_details:
          
          print ("found in bag: %s" % w)

  return(np.array(bag))

ERROR_THRESHOLD = 0.50
def classify(sentence):
  
  # generate probabilities from the model
  results = model.predict([bow(sentence, words)])[0]
  # filter out predictions below a threshold  
  results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
  
  # sort by strength of probability
  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in results:
    return_list.append((classes[r[0]], r[1]))
  # return tuple of intent and probability
  return return_list

def response(sentence, userID='123', show_details=False):
  
  results = classify(sentence)
  ##if we have no entents:--
  if results == []:
    
    return print("Sorry!!   Please try asking in a better way!!! or please contact 1800-110-200 for more details.!!")
    
  
  # if we have a classification then find the matching intent tag
  if results:
    # loop as long as there are matches to process
    while results:
      for i in intents['intents']:
        
        # find a tag matching the first result
        if i['tag'] == results[0][0]:
          
          # a random response from the intent
          return print(np.random.choice(i['responses']))
      results.pop(0)