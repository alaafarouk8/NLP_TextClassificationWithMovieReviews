# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:58:09 2021

@author: ALAA
"""
# path C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_TextClassificationWithMovieReviews\dataset
#import libraries
import re
import sklearn
import numpy as np
import numpy
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
import gensim
from sklearn import svm
from sklearn.neural_network import MLPClassifier
######################################################################################
# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = WordNetLemmatizer()
# Set values for various parameters
feature_size = 10    # Word vector dimensionality  
window_context = 10          # Context window size                                                                                    
min_word_count = 3   # Minimum word count                        
sample = 1e-3   # Downsample setting for frequent words
files = []
######################################################################################
# load files is done
def LoadFiles():
    dataset = sklearn.datasets.load_files(r"C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_TextClassificationWithMovieReviews\dataset", description=None, categories=None, load_content=True, shuffle=True, random_state=0)
    X , Y = dataset.data , dataset.target     # y is an array of 0s , 1s.
    return X , Y 
######################################################################################
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
         # tokenize the text into a list of words
         tokens = nltk.tokenize.word_tokenize(file)
         files.append(tokens)
    return files

#######################################################################################
#Averaging Words Vectors to Create Sentence Embedding
def get_mean_vector(word2vec_model, words):
    vocabulary = word2vec_model.wv.key_to_index
    words = [word for word in words if word in vocabulary]
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)
######################################################################################
#divide the data set randomly
def Splitingthedata(x,y):
    random_portion = round(np.random.rand(),3)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=random_portion, random_state=4)
    return Xtrain, Xtest, Ytrain, Ytest
#######################################################################################
#calling functions
x,y = LoadFiles()
files_=Preparedata(x)
print('Total training sentences: %d' % len(files_))
xtrain , xtest , ytrain , ytest = Splitingthedata(files_,y)
model =gensim.models.Word2Vec(files_, min_count=min_word_count,sample=sample)
# summarize vocabulary size in model
words = list(model.wv.key_to_index)
print('Vocabulary size: %d' % len(words))
def gettrain_array(xtrain):
    train_array=[]
    for i in xtrain:
        vec = get_mean_vector(model,i)
        if len(vec) > 0:
            train_array.append(vec)
    return train_array
def gettest_array(xtest):  
    test_array=[]     
    for i in xtest:
        vec = get_mean_vector(model,i)
        if len(vec) > 0:
            test_array.append(vec)
    return test_array
trainArr = gettrain_array(xtrain)  
testArr = gettest_array(xtest) 
classifier =svm.SVC(C=1.0, kernel='rbf', degree=3, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(trainArr, ytrain) 
yPredications = classifier.predict(testArr)
print("Accuracy: %" , accuracy_score(ytest, yPredications)*100)

f = input("enter review: ")

o = gettest_array([f])
output = classifier.predict(o)[0]
if (output ==1):
    print("positive review")
else:
    print("negative review")
#####################################################################################

