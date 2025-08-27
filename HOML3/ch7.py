import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch


from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.2)
vmodel = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True))
    ]
)
vmodel.fit(X, y)




from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1
)
bag.fit(X, y)




from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_features="sqrt", n_estimators=500, n_jobs=-1)
model.fit(X, y)






from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
model.fit(X, y)
#for score, name in zip(model.feature_importances_, iris.feature_names): print(name, score)




from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200, learning_rate=0.5
)
model.fit(X, y)




from sklearn.ensemble import GradientBoostingRegressor

m = 200
X = np.random.randn(m)
y = 3 * X * X + np.random.randn(m)

model = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
model.fit(X.reshape(-1, 1), y)







from sklearn.ensemble import StackingClassifier

model = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True))
    ],
    final_estimator=RandomForestClassifier(),
    cv=5
)
model.fit(X.reshape(-1, 1), y > 3 )
