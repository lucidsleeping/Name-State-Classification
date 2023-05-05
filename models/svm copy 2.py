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

dataset= pd.read_csv("dataset.csv")
array = dataset.values
# print(array.shape)
X = array[:,567:6201]
y = array[:,-1]
y=y.astype('int')

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=100)
dim1 = len(X_validation)
dim2 = len(X_validation[0:1])
print(dim1,dim2)

data = ["asdfgs"]

 
# normalised_data = []
# for i in data:
#     if len(i)>=4:
#                 i.replace(".","#")
#                 i.replace(" ","#")
#                 zero = 30 - len(i)
#                 for j in range(zero):
#                     i = i + "0"
#                 normalised_data.append(i)



model = Word2Vec(data, min_count = 1, vector_size = 5634)
vec = []

vec.append(model.wv[0])

vec = np.array(vec)
# print(len(vec),len(vec[0]))

models = []
models.append(('RF',RandomForestClassifier(n_estimators=100,max_depth=50))) #max acc

print(vec)

for name, model in models:
    model.fit(X,y)
    pred=model.predict(vec) #pred=[int(p) for p in pred]
    # pred=model.predict(X_validation) #pred=[int(p) for p in pred]
    print(pred)
    
            
#               precision    recall  f1-score   support

#            1       0.32      0.38      0.35        96
#            2       0.23      0.33      0.27        67
#            3       0.01      0.14      0.02         7
#            4       0.59      0.27      0.37       215

#     accuracy                           0.30       385
#    macro avg       0.29      0.28      0.25       385
# weighted avg       0.45      0.30      0.34       385


#####poly
#               precision    recall  f1-score   support

#            1       0.68      0.29      0.41       139
#            2       0.26      0.41      0.32        41
#            3       0.11      0.56      0.19         9
#            4       0.27      0.40      0.33        42

#     accuracy                           0.35       231
#    macro avg       0.33      0.42      0.31       231
# weighted avg       0.51      0.35      0.37       231