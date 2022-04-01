import streamlit as st
import base64
import pandas as pd
import pickle
import pathlib
def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl
first = "https://1drv.ms/x/s!AsazVgCmXrNWxmDxzZCuGdnRd-R1?e=h08E6s"
filenames = ["Fast0314", "Fast0314_2", "Normal0314", "Slow0314"]
files = ["https://1drv.ms/x/s!AsazVgCmXrNWxmDxzZCuGdnRd-R1?e=ThKRjA", "https://1drv.ms/x/s!AsazVgCmXrNWxmLMqAlpfwnPPvsd?e=RFQo5W"
        , "https://1drv.ms/x/s!AsazVgCmXrNWxmHSfwB0dloFY5Or?e=8fvGxf", "https://1drv.ms/x/s!AsazVgCmXrNWxmOJDDcTM6WeyBJT?e=bIMiMG"
        ]
st.sidebar.header('Data Selector')
st.sidebar.subheader('Select or upload the data you would like to view here')
uploaded_file = st.sidebar.file_uploader("Upload excel file", type=["xlsx", "csv"])

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
    df = pd.read_excel(create_onedrive_directdownload(files[filenames.index(selected)]))
    st.write(df)
st.write("""***""")
st.write("""Model loading""")
filename = st.text_input("Enter file name")

if filename is not "":
    loaded_model = pickle.load(open(filename, 'rb'))
    st.write("Model loaded")




