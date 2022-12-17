#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install tensorflow=
#!pip install tflearn

import nltk 

#nltk.download('punkt')

from nltk import word_tokenize,sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import random
import numpy as np 
import json
import tflearn
import pickle
import tensorflow as tf
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from colorama import Fore, Back, Style


# In[2]:


#ChatBot!!!
#it takes a couple of minutes to train

with open("graceintents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
               bag.append(1)
            else:
              bag.append(0)
    
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)


        
        
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch= 500, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch= 500, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)






# In[ ]:


# You can ask the bot 5 different types of questions about 13 different movies


def chat():
    
    movies =  "\n David Attenborough: A Life on Our Planet \n Inception \n Forrest Gump \n Anbe Sivam \n Bo Burnham: Inside \n Saving Private Ryan \n Django Unchained \n Dangal \n Bo Burnham: Make Happy \n Louis C.K.: Hilarious \n Dave Chappelle: Sticks & Stones \n 3 Idiots \n Black Friday \n Super Deluxe"
    print("Start talking with the bot! (type quit to stop) \n")
    print(Back.YELLOW + "Ask the bot about the movies:" )
          
    print(Style.RESET_ALL + movies + "\n \n")
    
    print(Back.YELLOW + "You can ask about the actors, directors, runtime, release date, or rating."  )
    print(Style.RESET_ALL + " \n \n \n")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]

        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))

        else:
            print("I didnt get that. Can you explain or try again.")
chat()


# In[ ]:





# In[ ]:





# In[ ]:




