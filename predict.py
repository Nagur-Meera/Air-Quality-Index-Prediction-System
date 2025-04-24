import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import os
from datetime import datetime

def load_models():
    """Load all trained models and scaler"""
    try:
        # Load models from the models directory
        linear_model = joblib.load('models/linear_model.joblib')
        rf_model = joblib.load('models/rf_model.joblib')
        nn_model = tf.keras.models.load_model('models/nn_model.keras')
        scaler = joblib.load('models/scaler.joblib')
        return linear_model, rf_model, nn_model, scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

def validate_input(feature, value):
    """Validate input values against predefined ranges"""
    ranges = {
        'T': (-50, 50),        # Temperature range in Celsius
        'RH': (0, 100),        # Relative Humidity range
        'Hour': (0, 23),       # Hour of day
        'NOx(GT)': (0, 1000),  # NOx range
        'NO2(GT)': (0, 1000),  # NO2 range
        'NMHC(GT)': (0, 1000), # NMHC range
        'C6H6(GT)': (0, 100)   # Benzene range
    }
    
    if feature in ranges:
        min_val, max_val = ranges[feature]
        if not (min_val <= value <= max_val):
            raise ValueError(f"{feature} must be between {min_val} and {max_val}")
    return True

def get_user_input():
    """Get input values from user with validation"""
    print("\nEnter the following air quality measurements:")

    features = {
        'PT08.S1(CO)': 'CO sensor (tin oxide) response',
        'PT08.S2(NMHC)': 'NMHC sensor (titania) response',
        'PT08.S3(NOx)': 'NOx sensor (tungsten oxide) response',
        'PT08.S4(NO2)': 'NO2 sensor (tungsten oxide) response',
        'PT08.S5(O3)': 'Ozone sensor (indium oxide) response',
        'T': 'Temperature in °C',
        'RH': 'Relative Humidity (%)',
        'C6H6(GT)': 'Benzene concentration in µg/m³',
        'NOx(GT)': 'NOx concentration in ppb',
        'NO2(GT)': 'NO2 concentration in µg/m³',
        'NMHC(GT)': 'Non Metanic HydroCarbons concentration in µg/m³'
    }

    input_data = {}
    for feature, description in features.items():
        while True:
            try:
                value = float(input(f"{description}: "))
                validate_input(feature, value)
                input_data[feature] = value
                break
            except ValueError as e:
                print(f"Error: {e}")
                print("Please enter a valid number")

    # Add current time-based features
    current_time = datetime.now()
    input_data['Hour'] = current_time.hour
    input_data['Month'] = current_time.month
    input_data['DayOfYear'] = current_time.timetuple().tm_yday

    # Calculate derived features
    input_data['NOx_NO2_ratio'] = input_data['NOx(GT)'] / (input_data['NO2(GT)'] + 1e-6)
    input_data['CO_NMHC_ratio'] = input_data['PT08.S1(CO)'] / (input_data['NMHC(GT)'] + 1e-6)

    # Create DataFrame with correct feature order
    feature_order = [
        'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)',
        'PT08.S5(O3)', 'T', 'RH', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'NMHC(GT)',
        'Hour', 'Month', 'DayOfYear', 'NOx_NO2_ratio', 'CO_NMHC_ratio'
    ]

    return pd.DataFrame([input_data])[feature_order]

def make_predictions(input_data, models, scaler):
    """Make predictions using all models with error handling"""
    try:
        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Make predictions
        linear_pred = models[0].predict(scaled_data)[0]
        rf_pred = models[1].predict(scaled_data)[0]
        nn_pred = models[2].predict(scaled_data, verbose=0)[0][0]

        return {
            'Linear Regression': linear_pred,
            'Random Forest': rf_pred,
            'Neural Network': nn_pred
        }
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def display_results(predictions):
    """Display prediction results with air quality categories"""
    print("\n" + "="*50)
    print("CO Level Predictions:")
    print("="*50)
    
    if predictions is None:
        print("Unable to make predictions due to an error")
        return

    for model, pred in predictions.items():
        print(f"{model}: {pred:.2f} mg/m³")

    # Calculate average prediction
    avg_pred = sum(predictions.values()) / len(predictions)
    print("\nEnsemble Average Prediction:", f"{avg_pred:.2f} mg/m³")

    # Display air quality category
    print("\nAir Quality Category:")
    if avg_pred < 1:
        print("Good: Air quality is considered satisfactory")
        print("Health implications: Little to no risk")
    elif avg_pred < 2:
        print("Moderate: Air quality is acceptable")
        print("Health implications: Some pollutants may affect very sensitive individuals")
    elif avg_pred < 3:
        print("Unhealthy for Sensitive Groups")
        print("Health implications: Children and people with respiratory conditions may experience health effects")
    elif avg_pred < 4:
        print("Unhealthy")
        print("Health implications: Everyone may begin to experience health effects")
    elif avg_pred < 5:
        print("Very Unhealthy")
        print("Health implications: Health warnings of emergency conditions")
    else:
        print("Hazardous: Health warnings of emergency conditions")
        print("Health implications: Everyone is more likely to be affected")

def main():
    print("="*50)
    print("Air Quality Index Prediction System")
    print("="*50)

    # Load models
    print("\nLoading trained models...")
    models = load_models()

    if None in models:
        print("Error: Could not load all required models.")
        print("Please ensure all model files exist in the 'models' directory.")
        return

    while True:
        try:
            # Get user input
            input_data = get_user_input()

            # Make predictions
            predictions = make_predictions(input_data, models, models[3])

            # Display results
            display_results(predictions)

            # Ask if user wants to make another prediction
            while True:
                another = input("\nWould you like to make another prediction? (y/n): ").lower()
                if another in ['y', 'n']:
                    break
                print("Please enter 'y' or 'n'")

            if another == 'n':
                break

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")

    print("\nThank you for using the Air Quality Index Prediction System!")

if __name__ == "__main__":
    main() 