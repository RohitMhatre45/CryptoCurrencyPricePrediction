import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import pandas_datareader as data
import datetime


# to run stream lit cmd :  py -m streamlit run app2.py

image = Image.open('6eawlkrg6xzxdcwn_1622439537.jpeg')
st.image(image,width=700)

st.title("CryptoCurrency Price Prediction")

start_date =  st.date_input(
     "Please enter start dates",datetime.date(2021, 4, 6))

end_date  = st.date_input(
     "Please enter End dates",datetime.date(2022, 4, 6))

option  = st.selectbox(
     'Select the CryptoCurrency ',
    ('ETH-USD', 'BTC-USD'))

# df = pd.read_csv("bitcon.csv")

df = data.DataReader(option,'yahoo',start_date,end_date)


if option == 'BTC-USD':
    st.subheader("Bitcoin Prices")
else:
    st.subheader("Etherium Prices")

if option == "BTC-USD":
    st.write(df.describe())
else:
    st.write(df.describe())

# st.write(df.describe())
if option == "BTC-USD":
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)
else:
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)


if option == "BTC-USD":
    st.subheader('Closing Price vs Time Chart With 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)
else:
    st.subheader('Closing Price vs Time Chart With 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)


if option == "BTC-USD":
    st.subheader('Closing Price vs Time Chart With 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig)
else:
    st.subheader('Closing Price vs Time Chart With 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig)

if option == "BTC-USD":

    data_Train = pd.DataFrame(df["Close"][0:int(len(df) * 0.70)])
    data_test = pd.DataFrame(df["Close"][int(len(df) * 0.70): int(len(df))])
else:
    data_Train = pd.DataFrame(df["Close"][0:int(len(df) * 0.70)])
    data_test = pd.DataFrame(df["Close"][int(len(df) * 0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_Train_array = scaler.fit_transform(data_Train)


model = load_model('keras_model.h5')


#testing Part

past_100_days = data_Train.tail(100)

final_df = past_100_days.append(data_test,ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)


# making predictons
#y_predicted = model.predict(x_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# #finalgraph
if option == "BTC-USD":
    st.subheader('Prediction vs Orignal')
    fig2 = plt.figure(figsize = (12,6))
    plt.plot(y_test,"b",label = "Orignal Price")
    plt.plot(y_predicted ,"r",label = "predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig2)
else:
    st.subheader('Prediction vs Orignal')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, "b", label="Orignal Price")
    plt.plot(y_predicted, "r", label="predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig2)



st.subheader('Orignal Values')
w = y_test
st.dataframe(w)

o = pd.DataFrame(w).to_csv()

st.download_button(
   "Press to Download orignal values",
   o,
   "file.csv",
   "text/csv",
   key='download-csv'
)


st.subheader('Predicted Values')
k = y_predicted
st.dataframe(k)

x = pd.DataFrame(k).to_csv()

st.download_button(
   "Press to Download the predicted Values",
   x,
   "fil.csv",
   "text/csv",
   key='download-csv')


st.markdown('<style>button{background: none;color:#ffa260;border: 2px solid;padding: 1em 2em;font-size: 1em;transition: color 0.25s, border-color 0.25s, box-shadow 0.25s,transform 0.25s;}button:hover{border-color: #f1ff5c; color: white;box-shadow: 0 0.5em 0.5em -0.4em #f1ff5c;transform: translateY(-0.25em);cursor: pointer;}</style><a href="https://docs.google.com/document/d/1B83w2d4t9tyLOFZPc9a8kCt_EVOMXyd1KSLJP62IF1I/edit?usp=sharing" target="_blank"><button>Google Docs</button></a>'
            ,unsafe_allow_html=True)


st.markdown('<br>',unsafe_allow_html=True)


st.markdown('<style>button{background: none;color:#ffa260;border: 2px solid;padding: 1em 2em;font-size: 1em;transition: color 0.25s, border-color 0.25s, box-shadow 0.25s,transform 0.25s;}button:hover{border-color: #f1ff5c; color: white;box-shadow: 0 0.5em 0.5em -0.4em #f1ff5c;transform: translateY(-0.25em);cursor: pointer;}</style><a href="https://docs.google.com/spreadsheets/d/1kE4t641NAd6VvsaOjoqhTmWahjybsbFjaHYJSxxUkUc/edit?usp=sharing" target="_blank"><button>Google Sheets</button></a>'
            ,unsafe_allow_html=True)
