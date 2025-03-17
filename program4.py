#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def CalculateRMSE():
    """
    This function loads the SO2TONS dataset, filters the data for the peak season 
    (May through August) and for Lake-1, trains a neural network model on the data, 
    and computes the RMSE error. The function returns a single numeric RMSE value.
    """
    # URL for the SO2TONS dataset
    url = "https://raw.githubusercontent.com/apownukepcc/ForecastingDailyEmissions/refs/heads/main/SO2TONS_dataset.csv"
    
    # Define the peak season months (May through August)
    peak_season_months = [5, 6, 7, 8]
    
    # The target lake
    target_source = "LAKE-1"
    
    # Load the dataset
    try:
        data = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading dataset from {url}: {e}")
        return None
    
    # Convert the 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Filter for peak season
    data = data[data['date'].dt.month.isin(peak_season_months)]
    
    # Filter for Lake-1 data only
    data = data[data['Source'] == target_source]
    
    # Check if the filtered data has enough rows
    if data.empty or len(data) < 10:
        print(f"Not enough data for SO2TONS at {target_source}.")
        return None
    
    # Define predictors and target variable
    predictors = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'pres']
    target = 'Emissions_Load'
    
    # Drop rows with missing values
    data = data.dropna(subset=predictors + [target])
    
    # Split the data into features (X) and target (y)
    X = data[predictors].values
    y = data[target].values
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build a simple neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    # Compile the model using mean squared error as the loss
    model.compile(optimizer='adam', loss='mse')
    
    # Set up early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop]
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Neural Network Model for SO2TONS at {target_source}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return rmse

if __name__ == "__main__":
    error = CalculateRMSE()
    if error is not None:
        print(f"\nCalculated RMSE error: {error:.4f}")
