from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import csv

# dataset= pd.read_csv("500.csv")
dataset= pd.read_csv("dataset.csv")
array = dataset.values
# print(array.shape)
X = array[:,567:6201]
y = array[:,-1]
y=y.astype('int')

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.50, random_state=100)

models = []
models.append(('SVM',svm.LinearSVC()))

scoring = {'accuracy' : make_scorer(accuracy_score),
       	'precision' : make_scorer(precision_score),
       	'recall' : make_scorer(recall_score),
       	'f1_score' : make_scorer(f1_score)}

for name, model in models:
    model.fit(X_train,Y_train)
    pred=model.predict(X_validation) #pred=[int(p) for p in pred]
    print(name)
    print(confusion_matrix(pred,Y_validation))
    print(classification_report(pred,Y_validation, zero_division=1))

            
#               precision    recall  f1-score   support

#            1       0.32      0.38      0.35        96
#            2       0.23      0.33      0.27        67
#            3       0.01      0.14      0.02         7
#            4       0.59      0.27      0.37       215

#     accuracy                           0.30       385
#    macro avg       0.29      0.28      0.25       385
# weighted avg       0.45      0.30      0.34       385
