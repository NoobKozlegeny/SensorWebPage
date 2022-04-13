import streamlit as st
import base64
import pandas as pd
import pickle
import pathlib
from pathlib import Path
import random
import numpy as np
import pandas as pd
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

#My classes
from Measurement import Measurement
from Test import Iris

def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl

# Visualizes accelerometer and gyroscope data
def visualize_data ():
    accelerometer_df = pd.read_csv(create_onedrive_directdownload(files[filenames.index(selected)]), skiprows=range(1,2))
    gyro_df = pd.read_csv(create_onedrive_directdownload(files[filenames.index(selected) + 1]), skiprows=range(1,2))

    st.write(f"This is {selected}'s data.")
    st.line_chart(accelerometer_df)
    st.write(f"This is {filenames[filenames.index(selected) + 1]}'s data.")
    st.line_chart(gyro_df)
    
#Runs classifications
def run_classifications(X, y):
    #Classifications
    model1 = DecisionTreeClassifier(random_state=2)
    model2 = KNeighborsClassifier(n_neighbors=50)
    model3 = LogisticRegression(random_state=2, max_iter=5000)
    vc = VotingClassifier(estimators=[('lr', model1), ('rf', model2), ('gnb', model3)], voting='hard')

    st.write("DecisionTreeClassifier:")
    st.line_chart(pd.DataFrame(data=stacking(model1, X, y)))
    st.write("KNeighborsClassifier:")
    st.line_chart(pd.DataFrame(data=stacking(model2, X, y)))
    st.write("LogisticRegression:")
    st.line_chart(pd.DataFrame(data=stacking(model3, X, y)))
    #stacking(model1, X, y)
    
    

def stacking(model,X,y):
    # split the Train dataset again
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=2)
    # fit the model on the NEW train dataset
    model.fit(X_train2, y_train2)
    # print the accuracy of the model on the NEW test dataset
    print(model.score(X_test2, y_test2))
    # return the model prediction of whole X dataset
    return model.predict(X)

first = "https://1drv.ms/u/s!Ao3uECOrt_zivFsWZaLIUT2MCYNd?e=3OSLQT"
filenames = ["FastACCELEROMETER0328", "FastGYROSCOPE0328",
             "NormalACCELEROMETER0328", "NormalGYROSCOPE0328"]
files = ["https://1drv.ms/u/s!Ao3uECOrt_zivFsWZaLIUT2MCYNd?e=3OSLQT", "https://1drv.ms/u/s!Ao3uECOrt_zivFzbxJbBqd6EHHzI?e=LdcSSy"
        , "https://1drv.ms/u/s!Ao3uECOrt_zivFqGnublBPKyWbg7?e=4OGr32", "https://1drv.ms/u/s!Ao3uECOrt_zivFmiE6GQ5w-PkeLG?e=aaB4lT"
        ]

st.sidebar.header('Data Selector')

st.sidebar.subheader('Select or upload the data you would like to view here')
uploaded_file = st.sidebar.file_uploader("Upload excel or pickle file", type=["xlsx", "csv", "pickle"])

selected = st.sidebar.selectbox('Select file by name', filenames)

st.write("""
# Data viewer

You can view the data you uploaded or the data you selected from the list here

""")

st.write("""***""")

if st.button('View uploaded'):
    if uploaded_file is not None:
        if pathlib.Path(uploaded_file.name).suffix == ".xlsx":
            udf = pd.read_excel(uploaded_file)
            st.write(udf)
        else: 
            cdf = pd.read_csv(uploaded_file)
            st.write(cdf)
    else: 
        st.write('No file was uploaded!')
        
st.write("""***""")

if st.button('View selected'):
    df = pd.read_csv(create_onedrive_directdownload(files[filenames.index(selected)]))
    st.write(df)
    
st.write("""***""")

st.write("""Model loading""")
filename = st.text_input("Enter file name")

if filename is not "":
    HERE = Path(__file__).parent
    loaded_model = pickle.load(open(HERE / filename, 'rb'))
    st.write("Model loaded")
    st.write(loaded_model)
    run_classifications(loaded_model.X, loaded_model.y)


#---------------------------------------------------------------------

st.write("""***""")

if st.button('Visualize data'):
    visualize_data()
    st.write("""***""")
    


