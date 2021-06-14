# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:58:09 2021

@author: ALAA
"""
# path C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_TextClassificationWithMovieReviews\dataset
#import libraries
import re
import nltk
import sklearn
import numpy as np
import numpy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics import plot_roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2vec
######################################################################################
# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = WordNetLemmatizer()
files = []
# load files is done
def LoadFiles():
    positive , negative = 0 , 0
    dataset = sklearn.datasets.load_files(r"C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_TextClassificationWithMovieReviews\dataset", description=None, categories=None, load_content=True, shuffle=True, random_state=0)
    X , Y = dataset.data , dataset.target     # y is an array of 0s , 1s.
    return X , Y 
# preparing data 
def  Preparedata(X):
    for i in range(len(X)):
         file = re.sub(r'\W', ' ', str(X[i]))
         file = re.sub(r'\s+[a-zA-Z]\s+', ' ', file)
         file = re.sub(r'\^[a-zA-Z]\s+', ' ', file) 
         file = re.sub(r'\s+', ' ', file, flags=re.I)
         file = re.sub(r'^b\s+', '', file)
         file = file.lower()
         file = file.split()
         file = [stemmer.lemmatize(word) for word in file]
         file = ' '.join(file)
         files.append(file)
    return files
x,y = LoadFiles()
files_=Preparedata(x)
print(len(files))
# Set values for various parameters of word2vec
num_features = 250  # Word vector dimensionality. Determines the no of words each word in the vocabulary will
#be associated with. Must be tuned.
min_word_count = 50   # Minimum word count. Wods occuring below the threshold will be ignored
num_workers = 1       # Number of threads to run in parallel
context = 15          # Context window size to be considered for each word                                             
downsampling = 1e-3   # Downsample setting for frequent words. To prevent more frequent words from dominating.

# Create Skip Gram model
model1= gensim.models.Word2Vec(files, min_count = 1,
                                             window = 5,)
# Print results
print("Cosine similarity between 'actor' " + 
               "and 'good' - CBOW : ",
    model1.similar_by_word('actor', 'good'))
