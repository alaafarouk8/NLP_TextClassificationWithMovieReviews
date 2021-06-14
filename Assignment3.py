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
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
import gensim
from gensim.models import Word2Vec
######################################################################################
# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = WordNetLemmatizer()
# Set values for various parameters
feature_size = 10    # Word vector dimensionality  
window_context = 10          # Context window size                                                                                    
min_word_count = 1   # Minimum word count                        
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
         files.append(file)
    return files

#######################################################################################
#Averaging Words Vectors to Create Sentence Embedding
def get_mean_vector(word2vec_model, words):
    vocabulary = word2vec_model.wv.key_to_index
    words = [(word) for word in words if word in vocabulary]
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)
######################################################################################
#divide the data set randomly
def Splitingthedata(x,y):
    random_portion = round(np.random.rand(),3)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=random_portion, random_state=0)
    return Xtrain, Xtest, Ytrain, Ytest
#######################################################################################
#calling functions
x,y = LoadFiles()
files_=Preparedata(x)
xtrain , xtest , ytrain , ytest = Splitingthedata(files_,y)
model =gensim.models.Word2Vec(files_, min_count=2,sample=sample)
train_array=[]
test_array=[]
def gettrain_array(xtrain):
    for i in xtrain:
        vec = get_mean_vector(model, i)
        if len(vec) > 0:
            train_array.append(vec)
    return train_array
def gettest_array(xtest):       
    for i in xtest:
        vec = get_mean_vector(model, i)
        if len(vec) > 0:
            test_array.append(vec)
    return test_array
trainArr = gettrain_array(xtrain)  
textArr = gettest_array(xtest) 
classifier = LogisticRegression()
classifier.fit(trainArr, ytrain) 
yPredications = classifier.predict(textArr)
print("Accuracy: %" , accuracy_score(ytest, yPredications)*100)
"""
f = input("enter review: ")
z=Preparedata(f)
o = gettest_array(z)
output = classifier.predict(o)
#####################################################################################
#count function 
def cnt(y_pred2):
    cntNeg = 0 
    cntPos = 0
    for i in range(len(y_pred2)):
        if (y_pred2[i]==0):
            cntNeg=cntNeg+1 
        else:
            cntPos=cntPos+1
    return cntNeg , cntPos 
###################################################
c1 , c2 = cnt(output)
if(c1>c2):
    print("negative")
else:
    print("positive")
    """