from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import svm

dataset= pd.read_csv("dataset.csv")
array = dataset.values
# print(array.shape)
X = array[:,567:6201]
y = array[:,-1]
y=y.astype('int')
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.40, random_state=100)


models = []
models.append(('RF',RandomForestClassifier(n_estimators=20,max_depth=100))) #max acc

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

#            1       0.80      0.36      0.50       111
#            2       0.00      1.00      0.00         0
#            3       0.00      1.00      0.00         0
#            4       0.23      0.19      0.21        43

#     accuracy                           0.31       154
#    macro avg       0.26      0.64      0.18       154
# weighted avg       0.64      0.31      0.42       154



            
