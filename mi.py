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
   
# #Runs classifications
# def run_classifications(X, y):
#     #Classifications
#     model1 = DecisionTreeClassifier(random_state=2)
#     model2 = KNeighborsClassifier(n_neighbors=50)
#     model3 = LogisticRegression(random_state=2, max_iter=5000)
#     vc = VotingClassifier(estimators=[('lr', model1), ('rf', model2), ('gnb', model3)], voting='hard')

#     st.write("DecisionTreeClassifier:")
#     st.line_chart(pd.DataFrame(data=stacking(model1, X, y)))
#     st.write(model1.predict(X))
    
#     st.write("KNeighborsClassifier:")
#     st.line_chart(pd.DataFrame(data=stacking(model2, X, y)))
#     st.write("DecisionTreeClassifier:")
    
#     st.write("LogisticRegression:")
#     st.line_chart(pd.DataFrame(data=stacking(model3, X, y)))
#     st.write("DecisionTreeClassifier:")
#     #stacking(model1, X, y)
    
    

# def stacking(model,X,y):
#     # split the Train dataset again
#     X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=2)
#     # fit the model on the NEW train dataset
#     model.fit(X_train2, y_train2)
#     # print the accuracy of the model on the NEW test dataset
#     st.write("accuracy:", model.score(X_test2, y_test2))
#     # return the model prediction of whole X dataset
#     return model.predict(X)

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
    X_train, X_test, y_train, y_test = train_test_split(loaded_model.X, loaded_model.y, test_size=0.2, random_state=2)
    
    #Running classification/predicting
    pred_DT = loaded_model.model_DT.predict(X_test)
    pred_KNN = loaded_model.model_KNN.predict(X_test)
    pred_LR = loaded_model.model_LR.predict(X_test)
    pred_V = loaded_model.model_V.predict(X_test)
    st.write("DecisionTreeClassifier accuracy: ", accuracy_score(y_test, pred_DT))
    st.write(pred_DT)
    plot_confusion_matrix(loaded_model.model_DT, X_test, y_test)
    st.pyplot()
    st.write("KNeighborsClassifier accuracy: ", accuracy_score(y_test, pred_KNN))
    st.write(pred_KNN)
    plot_confusion_matrix(loaded_model.model_KNN, X_test, y_test)
    st.pyplot()
    st.write("LogisticRegression accuracy: ", accuracy_score(y_test, pred_LR))
    st.write(pred_LR)
    plot_confusion_matrix(loaded_model.model_LR, X_test, y_test)
    st.pyplot()
    st.write("VotingClassifier accuracy: ", accuracy_score(y_test, pred_V))
    st.write(pred_V)
    plot_confusion_matrix(loaded_model.model_V, X_test, y_test)
    st.pyplot()
    # A plot attempt, looks weird
    # fig, ax = plt.subplots()
    # ax.plot(pred_DT, 'bo')
    # ax.plot(y_test, 'ro')
    # st.pyplot(fig)
    

#---------------------------------------------------------------------

st.write("""***""")

if st.button('Visualize data'):
    visualize_data()
    st.write("""***""")
    


