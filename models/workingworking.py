from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import csv

dataset= pd.read_csv("dataset.csv")
array = dataset.values
# print(array.shape)
X = array[:,567:6201]
y = array[:,-1]
y=y.astype('int')
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=100)
# dim1 = len(X_validation)
# dim2 = len(X_validation[0:1])
# print(dim1,dim2)


norm = [] 
inputvec = []
vecvec = []
vec = []

def normalization():
    data = [input()]
    for i in data:
        if len(i)>=4:
            i.replace(".","#")
            i.replace(" ","#")
            zero = 30 - len(i)
            for j in range(zero):
                i = i + "0"
            print(i)
            norm.append(i)
        else:
            continue

def data_creation():
    for data in norm:
        #appending name
        vec.append(data)
        data = list(data)
        model = Word2Vec(data, min_count = 1, vector_size = 200)
        for i in data:
            vec.append(list(model.wv[i]))
            #
            inputvec = []
            inputstring = ""
                
        vecvec = []
        for i in range(1,len(vec)):
            vec[i] = str(vec[i])
            vec[i] = vec[i].replace("[","")
            vec[i] = vec[i].replace("]","")
            x = vec[i].split(",")
            for j in x:
                inputvec.append(float(j))

        inputvec = np.array(inputvec[0:5634])
        vecvec.append(inputvec)
        vecvec = np.array(vecvec)

        return vecvec 
        
        
normalization()
vecvec = data_creation()

models = []
models.append(('RF',RandomForestClassifier(n_estimators=100,max_depth=50))) #max acc

print(vecvec)
for name, model in models:
    model.fit(X,y)
    pred=model.predict(vecvec) #pred=[int(p) for p in pred]
    print(pred)
    
            