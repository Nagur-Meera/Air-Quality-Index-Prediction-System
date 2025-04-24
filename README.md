# Air Quality Index Prediction System

## 🌐 Project Report
[View the full project report here](https://air-quality-index-prediction-system.netlify.app/)

## 📌 Project Overview
The **Air Quality Index (AQI) Prediction System** is a machine learning-based application developed to forecast **Carbon Monoxide (CO)** concentration in the atmosphere using environmental sensor data. The system utilizes an ensemble of models to provide reliable predictions and assess health impacts based on WHO standards.

## 🚀 Key Features
- Multi-model ensemble for accurate CO level prediction
- Real-time AQI assessment
- Health categorization based on WHO thresholds
- Interactive data visualizations
- Strong error handling and input validation mechanisms

## 📊 Dataset
- **Source**: UCI Air Quality Dataset
- **Time Period**: March 2004 to February 2005
- **Location**: An Italian city with significant pollution levels
- **Instances**: 9,358
- **Features**: 15 columns
- **Target**: CO(GT) — True CO concentration values

## 🧠 System Architecture
1. **Input Layer**: Captures sensor data
2. **Data Preprocessing**: Cleans and formats inputs
3. **Feature Engineering**: Scales and derives additional features
4. **Model Ensemble**: Combines outputs from Linear Regression, Random Forest, and Neural Network
5. **Prediction Layer**: Calculates final CO prediction
6. **Output Layer**: Maps prediction to health categories and implications

## 🔁 Data Flow Example
```json
Raw Input:
{
  "PT08.S1(CO)": "1360",
  "T": "13.6",
  "RH": "48.9",
  ...
}

Processed Input:
{
  "PT08.S1(CO)": 1360.0,
  "T": 13.6,
  "RH": 48.9,
  ...
}

Model Predictions:
{
  "linear": 2.81,
  "rf": 2.79,
  "nn": 2.70,
  "ensemble": 2.77
}

Final Output:
{
  "prediction": 2.77,
  "category": "Unhealthy for Sensitive Groups",
  "health_implications": "Children and people with respiratory conditions may experience health effects"
}
