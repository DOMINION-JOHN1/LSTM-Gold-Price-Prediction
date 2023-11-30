import yfinance as yf
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 2: Acquire historical data from Yahoo Finance
def fetch_gold_data():
    gold_data = yf.download('GC=F', period='1y')
    return gold_data

# Step 3: Preprocess and clean the data
def preprocess_data(data):
    data = pd.DataFrame(data)
    # Convert the "Date" column to a datetime object
    data.index = pd.to_datetime(data.index)
    # Create a new column for the month of the year (as numbers)
    data['Month_Num'] = data.index.month
    # Create a new column for the day of the week (as numbers, where Monday is 0 and Sunday is 6)
    data['Day_of_Week_Num'] = data.index.dayofweek
    data = data.reset_index(drop=True).drop(columns=['Adj Close'])
    data['Next Close'] = data['Close'].shift(-1)
    return data

# Step 4: Update and retrain your LSTM model
def update_and_retrain_model(model, scaler, data):
    # Extract features and target variable
    X = data.drop(['Next Close'], axis=1)
    y = data['Next Close']

    # Handle NaN values in the target variable for sequence prediction
    nan_indices = np.isnan(y)
    y = y.fillna(0)  # Fill NaN values with a placeholder (you can choose a different value)

    # Standardize features
    X = scaler.fit_transform(X)

    # Reshape X for LSTM input
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # Train the model
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

    return model


# Main function
def main():
    scaler = StandardScaler()

    # Load the existing LSTM model
    model = keras.models.load_model('best_model (1).h5')

    # Step 1: Set up the script
    print("Updating the LSTM model with the latest data...")

    # Step 2: Acquire the latest historical data from Yahoo Finance
    gold_data = fetch_gold_data()

    # Step 3: Preprocess and clean the data
    cleaned_data = preprocess_data(gold_data)

    # Step 4: Update and retrain your LSTM model
    updated_model = update_and_retrain_model(model, scaler, cleaned_data)

    # Save the updated LSTM model
    updated_model.save('best_model (1).h5')

if __name__ == '__main__':
    main()
