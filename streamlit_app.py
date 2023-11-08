import streamlit as st
import joblib

# Load the updated machine learning model
model = joblib.load('GOLDH&L(2).h5')

# Streamlit app
def main():
    st.title('Gold Price Predictor')

    st.write("Welcome to the Gold Price Predictor App!")

    # User interaction and data input
    st.sidebar.header('User Input:')
    open_price = st.sidebar.number_input('Enter the Open Price:', min_value=0.0)
    high_price = st.sidebar.number_input('Enter the High Price:', min_value=0.0)
    low_price = st.sidebar.number_input('Enter the Low Price:', min_value=0.0)
    close_price = st.sidebar.number_input('Enter the Close Price:', min_value=0.0)
    volume = st.sidebar.number_input('Enter the Volume:', min_value=0.0)
    day_of_month = st.sidebar.number_input('Enter the Day of the Month:', min_value=1, max_value=31)
    day_of_week = st.sidebar.number_input('Enter the Day of the Week (1=Monday, 7=Sunday):', min_value=1, max_value=7)

    if st.sidebar.button('Predict'):
        # Use the loaded model to make predictions
        input_data = [[open_price, high_price, low_price, close_price, volume, day_of_month, day_of_week]]
        prediction = model.predict(input_data)[0]

        # Calculate the next high and low
        next_high = prediction[0]
        next_low = prediction[1]

        st.write(f'Predicted Next High Price: {next_high:.2f}')
        st.write(f'Predicted Next Low Price: {next_low:.2f}')

    # Display model information
    st.sidebar.header('Model Information:')
    st.sidebar.write("This model has been updated with the latest data.")

if __name__ == '__main__':
    main()
