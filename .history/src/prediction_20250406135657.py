"""
Module for data preprocessing functions.
"""

def preprocess_data(data):
    """
    Preprocess the input data.
    
    Args:
        data (DataFrame): The raw data to preprocess.
    
    Returns:
        DataFrame: The preprocessed data.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pandas_datareader as data
    import yfinance as yf
    import streamlit as st
    from keras.models import load_model


    start = '2010-01-01'
    end = '2023-12-7'

    st.title("Stock Trend Prediction")

    user_input = st.text_input('Enter Stock Ticker', 'AAPL')
    df = yf.download(user_input, start=start, end=end)

    #Describing Data
    st.subheader('Data from 2010-2023')
    st.write(df.describe())

    #Visulatizations
    st.subheader('Closing Price vs Yime chart ')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Yime chart with 100ma')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100ma & 200ma')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)

    # Load my model
    model = load_model('keras_model.h5')

    #Testing part

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i,0])
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    scaler = scaler.scale_

    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor


    # Final graph
    st.subheader('Predictions vs Original')
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, 'b', label='Original Price')
    ax.plot(y_predicted , 'r' , label='Predicted Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig2)

    # Clear the Matplotlib figure to avoid potential issues
    plt.close(fig2)

        pass