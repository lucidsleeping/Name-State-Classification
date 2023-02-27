from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier 
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier


dataset= pd.read_csv("dataset.csv")
array = dataset.values
print(array.shape)
X = array[:,567:6201]
y = array[:,-1]
y=y.astype('int')
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.40, random_state=100)

models = []
models.append(('Aboost',AdaBoostClassifier(n_estimators=100,random_state = 0)))
        

scoring = {'accuracy' : make_scorer(accuracy_score),
            'precision' : make_scorer(precision_score),
            'recall' : make_scorer(recall_score),
            'f1_score' : make_scorer(f1_score)}

for name, model in models:
    model.fit(X_train,Y_train)
    pred=model.predict(X_validation) 
    print(name)
    print(confusion_matrix(pred,Y_validation))
    print(classification_report(pred,Y_validation, zero_division=1))




            
#               precision    recall  f1-score   support

#            1       0.42      0.41      0.42        51
#            2       0.51      0.29      0.37        63
#            3       0.12      0.29      0.17        14
#            4       0.17      0.23      0.20        26

#     accuracy                           0.32       154
#    macro avg       0.31      0.30      0.29       154
# weighted avg       0.39      0.32      0.34       154