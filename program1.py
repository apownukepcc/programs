#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def CalculateRMSE():
    """
    This function loads the SO2TONS dataset, filters the data for the peak season 
    (May through August) and for Lake-1, trains a Random Forest Regressor on the data, 
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
    X = data[predictors]
    y = data[target]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model for SO2TONS at {target_source}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return rmse

if __name__ == "__main__":
    error = CalculateRMSE()
    if error is not None:
        print(f"\nCalculated RMSE error: {error:.4f}")
