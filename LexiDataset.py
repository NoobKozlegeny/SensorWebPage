import pickle
from sklearn import datasets
import pandas as pd
import os
import random
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score


def FittingAndEvaulate(model,X,y):
    # split the Train dataset again
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    # fit the model on the NEW train dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print the accuracy of the model on the NEW test dataset
    print("accuracy:", accuracy_score(y_test, y_pred))

# Lexi's code UwU
class WallClimbing():
    # Reading the files
    ACCFast = pd.read_csv("FastACCELEROMETER0407.csv")
    ACCSlow = pd.read_csv("SlowACCELEROMETER0407.csv")
    ACCNorm = pd.read_csv("NormalACCELEROMETER0408.csv")
    
    # Getting the timestamps from the data
    FastTS=ACCFast.iloc[[0]]
    SlowTS=ACCSlow.iloc[[0]]
    NormTS=ACCNorm.iloc[[0]]

    # Creating dataframe and naming the labels
    AllData1 = ACCFast[ACCFast['X'] != 'Timestamp: ']
    AllData2 = ACCNorm[ACCNorm['X'] != 'Timestamp: ']
    AllData3 = ACCSlow[ACCSlow['X'] != 'Timestamp: ']
    AllData = pd.concat([AllData1, AllData2, AllData3], ignore_index=True, sort=False)

    # Separating feature/data and target
    X = AllData.iloc[:,:3]
    y = AllData.iloc[:,-1]

    # Creating the modells
    modelDTC = DecisionTreeClassifier(criterion ='entropy', max_depth =2, random_state=69)
    modelSVC = SVC(kernel='linear')
    param_grid = {'C': [0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 0.001],
                'kernel': ['rbf']}
    modelGridSearchCV = GridSearchCV(SVC(), param_grid, verbose = 2)
    
    #Fitting/training models
    FittingAndEvaulate(modelDTC, X, y)
    FittingAndEvaulate(modelSVC, X, y)
    FittingAndEvaulate(modelGridSearchCV, X, y)
    
    #print(modelGridSearchCV.best_params_)
    
#Pickling models
instWallClimbing = WallClimbing()
print(os.getcwd())

pickle.dump(instWallClimbing,
            open('wallClimbing.pickle', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)
