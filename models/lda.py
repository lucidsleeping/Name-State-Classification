from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

dataset= pd.read_csv("dataset.csv")
array = dataset.values
# print(array.shape)
X = array[:,567:6201]
y = array[:,-1]
y=y.astype('int')


X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=100)
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
scoring = {'accuracy' : make_scorer(accuracy_score),
         'precision' : make_scorer(precision_score),
         'recall' : make_scorer(recall_score),
         'f1_score' : make_scorer(f1_score)}

for name, model in models:
    model.fit(X_train,Y_train)
    pred=model.predict(X_validation)
    print(name)
    pred=[int(p) for p in pred]
    print(confusion_matrix(pred,Y_validation))
    print(classification_report(pred,Y_validation, zero_division=1))

#               precision    recall  f1-score   support

#            1       0.00      1.00      0.00         0
#            2       1.00      0.23      0.37       153
#            3       0.00      1.00      0.00         0
#            4       0.00      0.00      0.00         1

#     accuracy                           0.23       154
#    macro avg       0.25      0.56      0.09       154
# weighted avg       0.99      0.23      0.37       154
