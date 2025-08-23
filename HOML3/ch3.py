import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch



from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target
print(X.shape, y.shape)

Xtrain, ytrain = X[:60000], y[:60000]
Xtest, ytest = X[60000:], y[60000:]


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

ytrainfive = (ytrain == '5')
ytestfive = (ytest == '5')
clf = SGDClassifier(random_state=42)  ### Commented out to save time
clf.fit(Xtrain, ytrainfive)
ypredfive = clf.predict(Xtrain)
cm = confusion_matrix(ytrainfive, ypredfive)
print(f"Precision: {precision_score(ypredfive, ytrainfive)}, Recall: {recall_score(ypredfive, ytrainfive)}, F1score: {f1_score(ypredfive, ytrainfive)}")



score = cross_val_score(clf, Xtrain, ytrainfive, cv=3, scoring="accuracy")
ypredfive = cross_val_predict(clf, Xtrain, ytrainfive, cv=3)
cm = confusion_matrix(ytrainfive, ypredfive)
print(f"Precision: {precision_score(ypredfive, ytrainfive)}, Recall: {recall_score(ypredfive, ytrainfive)}, F1score: {f1_score(ypredfive, ytrainfive)}")
clf.decision_function([X[1]])
precisions, recalls, thresholds = precision_recall_curve(ypredfive, ytrainfive)





from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(Xtrain, ytrainfive)
ypredfive = rfc.predict(Xtrain)
cm = confusion_matrix(ytrainfive, ypredfive)
print(cm)





from sklearn.svm import SVC
svc = SVC()
svc.fit(X[:1000], y[:1000])
svc.predict([X[0]])
#print(svc.classes_)
#print(svc.decision_function([X[0]]))


from sklearn.multiclass import OneVsRestClassifier
ovrc = OneVsRestClassifier(SVC())
ovrc.fit(Xtrain[:1000], ytrain[:1000])
ovrc.predict([X[0]])



from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier()
sgdc.fit(Xtrain, ytrain)
sgdc.predict([X[0, :]])



from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

ss = StandardScaler()
Xtt = ss.fit_transform(Xtrain.astype("float64"))
#ypred = sgdc.predict(Xtt)
ypred = cross_val_predict(sgdc, Xtt, ytrain, cv=3)
ConfusionMatrixDisplay.from_predictions(ytrain, ypred)
