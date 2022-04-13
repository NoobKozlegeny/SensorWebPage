import pickle
from sklearn import datasets
import pandas as pd
import os

#Creating models to pickle later
class Iris():
    iris = datasets.load_iris(return_X_y=True, as_frame=True)
    X = iris[0] #data
    y = iris[1] #target

    print(X)
    print(y)

#Pickling models
instIris = Iris()
print(os.getcwd())

pickle.dump(instIris,
            open('iris.pickle', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)