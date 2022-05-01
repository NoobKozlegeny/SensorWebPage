import pylint
import streamlit as st
import base64
import pandas as pd
import pickle
import pathlib
import random
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

#My classes
from Measurement import Measurement
from Test import Iris
from LexiDataset import WallClimbing

st.set_option('deprecation.showPyplotGlobalUse', False)

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
    
    #Getting test data
    X_train, X_test, y_train, y_test = train_test_split(loaded_model.X, loaded_model.y, test_size=0.2, random_state=69)
    
    #Running classification/predicting
    # pred_DT = loaded_model.model_DT.predict(X_test)
    # pred_KNN = loaded_model.model_KNN.predict(X_test)
    # pred_LR = loaded_model.model_LR.predict(X_test)
    # pred_V = loaded_model.model_V.predict(X_test)
    # st.write("DecisionTreeClassifier accuracy: ", accuracy_score(y_test, pred_DT))
    # st.write(pred_DT)
    # plot_confusion_matrix(loaded_model.model_DT, X_test, y_test)
    # st.pyplot()
    # st.write("KNeighborsClassifier accuracy: ", accuracy_score(y_test, pred_KNN))
    # st.write(pred_KNN)
    # plot_confusion_matrix(loaded_model.model_KNN, X_test, y_test)
    # st.pyplot()
    # st.write("LogisticRegression accuracy: ", accuracy_score(y_test, pred_LR))
    # st.write(pred_LR)
    # plot_confusion_matrix(loaded_model.model_LR, X_test, y_test)
    # st.pyplot()
    # st.write("VotingClassifier accuracy: ", accuracy_score(y_test, pred_V))
    # st.write(pred_V)
    # plot_confusion_matrix(loaded_model.model_V, X_test, y_test)
    # st.pyplot()
    
    # Wall Climbing
    pred_DTC = loaded_model.modelDTC.predict(X_test)
    pred_SVC = loaded_model.modelSVC.predict(X_test)
    pred_GridSearchCV = loaded_model.modelGridSearchCV.predict(X_test)
    
    st.write("DecisionTreeClassifier accuracy: ", accuracy_score(y_test, pred_DTC))
    st.write(pred_DTC)
    plot_confusion_matrix(loaded_model.modelDTC, X_test, y_test)
    st.pyplot()
    st.write("SVC accuracy: ", accuracy_score(y_test, pred_SVC))
    st.write(pred_SVC)
    plot_confusion_matrix(loaded_model.modelSVC, X_test, y_test)
    st.pyplot()
    st.write("GridSearchCV accuracy: ", accuracy_score(y_test, pred_GridSearchCV))
    st.write(pred_GridSearchCV)
    plot_confusion_matrix(loaded_model.modelGridSearchCV, X_test, y_test)
    st.pyplot()

#---------------------------------------------------------------------

st.write("""***""")

if st.button('Visualize data'):
    visualize_data()
    st.write("""***""")
    


