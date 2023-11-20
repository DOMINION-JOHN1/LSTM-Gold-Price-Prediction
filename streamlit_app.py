import pandas as pd
import streamlit as st
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load the updated machine learning model
model = keras.models.load_model('best_model (1).h5')
scaler = StandardScaler()

# Decorate the prediction function with tf.function
@tf.function(reduce_retracing=True)
def predict_price(Open, High, Low, Close, Volume, Month_Num, Day_of_Week_Num):
    # Use the loaded model to make predictions
    input_data = np.array([[Open, High, Low, Close, Volume, Month_Num, Day_of_Week_Num]])
    column_names = ["Open", "High", "Low", "Close", "Volume", "Month_Num", "Day_of_Week_Num"]
    # Creating a DataFrame
    input_data = pd.DataFrame(data=input_data, columns=column_names)
    input_data = scaler.fit_transform(input_data)
    input_data = input_data.reshape(input_data.shape[0], 1, 7)
    prediction = model.predict(input_data)

    # Calculate the next close (assuming it's the first output in the prediction)
    next_close = prediction[0][0]

    return next_close

# Streamlit app
def main():
    st.title('Gold Price Predictor')

    st.write("Welcome to the Gold Price Predictor App!")

    image = Image.open("gold5.png")
    st.image(image, use_column_width=True)

    # User interaction and data input
    st.sidebar.header('User Input:')
    Open = st.sidebar.number_input('Enter the Open Price:', min_value=0.0)
    High = st.sidebar.number_input('Enter the High Price:', min_value=0.0)
    Low = st.sidebar.number_input('Enter the Low Price:', min_value=0.0)
    Close = st.sidebar.number_input('Enter the Close Price:', min_value=0.0)
    Volume = st.sidebar.number_input('Enter the Volume:', min_value=0.0)
    Month_Num = st.sidebar.number_input('Enter the Day of the Month:', min_value=1, max_value=31)
    Day_of_Week_Num = st.sidebar.number_input('Enter the Day of the Week (1=Monday, 7=Sunday):', min_value=1, max_value=7)

    if st.sidebar.button('Predict'):
        # Call the decorated function
        next_close = predict_price(Open, High, Low, Close, Volume, Month_Num, Day_of_Week_Num)

        st.write(f'Predicted Next Close Price: {next_close:.2f}')

    # Display model information
    st.sidebar.header('Model Information:')
    st.sidebar.write("This model has been updated with the latest data.")

if __name__ == '__main__':
    main()
