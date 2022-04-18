import pickle
from sklearn import datasets
import pandas as pd
import os
import random
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_digits


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score

#Creating models to pickle later
def stacking(model,X,y):
    # split the Train dataset again
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=2)
    # fit the model on the NEW train dataset
    model.fit(X_train2, y_train2)
    # print the accuracy of the model on the NEW test dataset
    print("accuracy:", model.score(X_test2, y_test2))
    # # return the model prediction of whole X dataset
    # return model.predict(X)
class Iris():
    iris = datasets.load_iris(return_X_y=True, as_frame=True)
    X = iris[0] #data
    y = iris[1] #target
    
    #Classifications
    model_DT = DecisionTreeClassifier(random_state=2)
    model_KNN = KNeighborsClassifier(n_neighbors=50)
    model_LR = LogisticRegression(random_state=2, max_iter=5000)
    model_V = VotingClassifier(estimators=[('dtc', model_KNN), ('knn', model_DT), ('lr', model_LR)], voting='hard')
    
    #Fitting/training models
    stacking(model_DT, X, y)
    stacking(model_KNN, X, y)
    stacking(model_LR, X, y)
    stacking(model_V, X, y)

#Pickling models
instIris = Iris()
print(os.getcwd())

pickle.dump(instIris,
            open('iris.pickle', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)