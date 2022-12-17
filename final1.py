#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import json
import opendatasets as od
import pymongo
import csv
import nltk 
#nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import random
import numpy as np 
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# uploading CSV files using pandas dataframes

exists1 = os.path.exists("Best Movies Netflix.csv")
exists2 = os.path.exists("raw_credits.csv")
exists3 = os.path.exists("raw_titles.csv")


if exists1 == True:
    file1 = pd.read_csv("Best Movies Netflix.csv")
else:
    print("The file does not exist.  Please download the CSV file.") 
    
if exists2 == True:
    file2 = pd.read_csv('raw_credits.csv')
else :
    print("The file does not exist.  Please download the CSV file.")
    
    
if exists3 == True:
    file3 = pd.read_csv('raw_titles.csv')
else :
    print("The file does not exist.  Please download the CSV file.")


# In[3]:


# Selecting Useful Data from raw titles

newdf = file3[file3['type'] == 'MOVIE']
rawtitles= newdf [['id', 'title', 'runtime', 'genres']]


# In[4]:


# select only the top 13 movies ( so the dataset is small enough that the program will run)

file1 = file1[file1["SCORE"] > 8.3 ]


# In[5]:


#filter out only movies from rawtitles that are best movie list


mwanted1 = file1['TITLE'].tolist()

titles = pd.DataFrame()

for title in mwanted1:
    x = rawtitles.loc[rawtitles['title'] == title]
    titles = titles.append(x)


# In[6]:


# filter out only actors that are in best movie list from raw credits

awanted  = titles['id'].tolist()

credits = pd.DataFrame()

for value in awanted:
    y = file2.loc[file2['id'] == value]
    credits = credits.append(y)
    


# In[7]:


#Inserting Data into Mongodb (file1, titles, credits)

md = file1.to_dict("records")
md1 = titles.to_dict("records")
md2  = credits.to_dict("records")

myclient = pymongo.MongoClient()


dbnames = myclient.list_database_names()
if 'Netflix' in dbnames:
    mydb = myclient["Netflix"]
    mycol = mydb["movies"]
    mycol1 = mydb["titles"]
    mycol2 = mydb["credits"]
else:
    mydb = myclient["Netflix"]
    mycol = mydb["movies"]
    mycol1 = mydb["titles"]
    mycol2 = mydb["credits"]
    mydb.movies.insert_many(md)
    mydb.titles.insert_many(md1)
    mydb.credits.insert_many(md2)
    
    


# In[8]:


# get actors 



def get_actors(movie):
    
    actors = " "
    
    
    filter = {"title": movie}
    cursor = mydb.titles.find(filter)
    
    
    for x in cursor:
        ID = x.get('id')
        
    filter1 = {"id": ID, "role" : "ACTOR"}
    cursor1 = mydb.credits.find(filter1)
    

    for y in cursor1:
         actors += y.get('name')  + "\n"
        
     
    return actors


# In[9]:


#get director



def get_director(movie):
    
    
    director = ""
 
    filter = {"title": movie}
    cursor = mydb.titles.find(filter)
    
    
    for x in cursor:
        ID = x.get('id')
        
        filter1 = {"id": ID, "role" : "DIRECTOR"}
        cursor1 = mydb.credits.find(filter1)
    

    for y in cursor1:
        director += y.get('name') + "\n"
        
    return director
        


# In[10]:


# get rating


def get_rating(movie):
    
    rating = " "
    
    filter = {"TITLE": movie}
    
    cursor = mydb.movies.find(filter)
    
    for x in cursor:
        y = str( x.get("SCORE"))

    return rating + y + " out of 10, according to IMDB voters "
    
    
    
    


# In[11]:


# get runtime

def get_runtime(movie):
    
    
    runtime = ""
    filter = {"TITLE": movie}
    
    cursor = mydb.movies.find(filter)
    
    for x in cursor:
        y = str(x.get("DURATION"))
        
    runtime += y + " " + "minutes"

    return runtime
    


# In[12]:


# get year

def get_year(movie):
    
    year = " "
    
    filter = {"TITLE": movie}
    
    cursor = mydb.movies.find(filter)
    for x in cursor:
        y = str(x.get("RELEASE_YEAR"))
        
    return year + y
    


# In[13]:


#Making intents file


data = {'intents':[ { 'tag' : "greeting", 'patterns' : ["Hi", "How are you", "Is anyone there?", "Hello", "Hey","Good day", "Whats up","Hola"],
            'responses' : ["Hello!", "Good to see you again!", "Hi there, how can I help?","hurry up, I don't have all day"],
            'context set':''},
                   
            {"tag": "goodbye",
         "patterns": ["cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day","bye"],
         "responses": ["Sad to see you go..", "Talk to you later", "Goodbye!"],
         "context_set": ""},
                   
             {'tag': 'Movie recommendation', 'patterns' : ['What is a good movie?', 'Recommend a movie.', "What movie do you like?" , "What's the best movie?", "Which is the best movie?", "Tell me a good movie."],
             'responses' : ["What genre of movie do you like? Options: ", "What type of movie are you looking for?"],
             'context set': ''},
                   
                   ] }      




datm = file1['TITLE']
            

#adding movie actors directors


for movie in datm:
    
    case = {"tag" : movie, 'patterns' : ["Who acts in "+ movie, "Who is acting in "+ movie, "What actors are in "+ movie], 'responses' : ["This is the list: " + get_actors(movie), "Here you go :" + get_actors(movie)], 'context set': ''}
    
    case1 = {"tag" : movie + " director", "patterns" : ["Who directs "+ movie, "Who is the director of "+ movie, "Do you know who directs " + movie], 'responses': ["Here you go : " + get_director(movie), "This is the director : " + get_director(movie)], 'context set': ''}
    
    case2 = {"tag" : movie + " runtime", "patterns" : ["What is the runtime of  "+ movie, "How long is "+ movie, "Do you know the runtime of " + movie], 'responses': ["Here you go : " + get_runtime(movie), "This is the runtime : " + get_runtime(movie)], 'context set': ''}
    
    case3 = {"tag" : movie + " year", "patterns" : ["What year was  "+ movie + "released", "When was "+ movie + "released", "When did they make " + movie], 'responses': ["Here you go : " + get_year(movie), "This is the year : " + get_year(movie)], 'context set': ''}
    
    
    case4 = {"tag" : movie + "rating", "patterns" : ["What was the rating of "+ movie , "How was "+ movie + "rated", "How good is " + movie], 'responses': ["Here you go : " + get_rating(movie), "This is the rating : " + get_rating(movie)], 'context set': ''}
    
    
    
    data["intents"].append(case1)
    data["intents"].append(case)
    data["intents"].append(case2)
    data["intents"].append(case3)
    data["intents"].append(case4)
    


# In[14]:


if os.path.exists('graceintents.json')  == False :
    with open('graceintents.json', 'w') as f:
        json.dump(data, f, indent = 2)  


# In[ ]:




