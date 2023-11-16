!pip install yfinance
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import RobustScaler


# Step 2: Acquire historical data from Yahoo Finance
def fetch_gold_data():
    gold_data = yf.download('Gold', period='1d')
    return gold_data

# Step 3: Preprocess and clean the data
def preprocess_data(data):
    data=pd.DataFrame(data)
    # Convert the "Date" column to a datetime object
    data.index=pd.to_datetime(data.index)
    # Create a new column for the month of the year (as numbers)
    data['Month_Num']=data.index.month
    # Create a new column for the day of the week (as numbers, where Monday is 0 and Sunday is 6)
    data['Day_of_Week_Num']=data.index.dayofweek
    data = data.reset_index(drop=True).drop(columns=['Adj Close'])
    data['Next High']=data['High'].shift(-1)
    data['Next Low']=data['Low'].shift(-1)
    return data

# Step 4: Update and retrain your LSTM model
def update_and_retrain_model(model, scaler, data):
    data.dropna(inplace=True)
    X = data.drop(['Next High', 'Next Close', 'Next Low'], axis=1)
    y = data[['Next High', 'Next Low']]
    X = scaler.fit_transform(X)
    X = X (X[0], 1, 7)
    model.fit(X, y, epochs=34, batch_size=32, validation_split=0.2)
    return model

# Main function
def main():

    scaler = RobustScaler()
    # Load the existing LSTM model
    model = keras.models.load_model('GOLDH&L (2).h5')

    # Step 1: Set up the script
    print("Updating the LSTM model with the latest data...")

    # Step 2: Acquire the latest historical data from Yahoo Finance
    gold_data = fetch_gold_data()

    # Step 3: Preprocess and clean the data
    cleaned_data = preprocess_data(gold_data)

    # Step 4: Update and retrain your LSTM model
    updated_model = update_and_retrain_model(model, scaler, cleaned_data)

    # Save the updated LSTM model
    updated_model.save('GOLDH&L.h5')

if __name__ == '__main__':
    main()
