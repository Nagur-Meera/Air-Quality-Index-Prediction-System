import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
import joblib
import os
warnings.filterwarnings('ignore')

def print_section(title, description=""):
    print("\n" + "="*100)
    print(f"{title:^100}")
    if description:
        print(f"{description:^100}")
    print("="*100)

def print_subsection(title, description=""):
    print("\n" + "-"*50)
    print(f"{title}")
    if description:
        print(f"{description}")
    print("-"*50)

# Step 1: Data Loading and Initial Exploration
def load_and_explore_data():
    print_section("STEP 1: DATA LOADING AND EXPLORATION",
                 "Analyzing the Air Quality dataset structure and characteristics")

    print("\nLoading the Air Quality dataset...")
    df = pd.read_excel('AirQualityUCI.xlsx')

    print_subsection("1.1 Dataset Overview", "Basic information about the dataset")
    print(f"\nDataset Dimensions:")
    print(f"Number of rows (samples): {df.shape[0]:,}")
    print(f"Number of columns (features): {df.shape[1]}")

    print("\nFirst 5 samples of the dataset:")
    print(df.head().to_string())

    print_subsection("1.2 Data Types and Missing Values Analysis",
                    "Understanding data types and checking for missing values")
    print("\nData Types:")
    print(df.dtypes.to_string())

    print("\nMissing Values Analysis:")
    missing_stats = df.isnull().sum()
    missing_percent = (missing_stats / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Count': missing_stats,
        'Missing Percentage': missing_percent
    })
    print(missing_info.to_string())

    print_subsection("1.3 Statistical Summary",
                    "Descriptive statistics of numerical features")
    print("\nBasic Statistics:")
    df_corr = df.copy()
    
    # Convert Date and Time columns properly
    df_corr['Date'] = pd.to_datetime(df_corr['Date'])
    # Extract hour from Time column (assuming it's in HH:MM:SS format)
    df_corr['Time'] = pd.to_datetime(df_corr['Time'].astype(str)).dt.hour

    # Select only numerical columns for correlation
    numerical_cols = df_corr.select_dtypes(include=[np.number]).columns
    print(df_corr[numerical_cols].describe().to_string())

    print("\nFeature Correlation with CO(GT):")
    correlations = df_corr[numerical_cols].corr()['CO(GT)'].sort_values(ascending=False)
    print(correlations.to_string())

    return df

# Step 2: Data Preprocessing
def preprocess_data(df):
    print_section("STEP 2: DATA PREPROCESSING",
                 "Cleaning and preparing data for model training")

    print_subsection("2.1 Missing Values Handling",
                    "Processing missing values marked as -200")
    initial_rows = len(df)

    df_clean = df.copy()
    df_clean = df_clean.replace(-200, np.nan)
    
    imputer = KNNImputer(n_neighbors=5)
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numerical_cols] = imputer.fit_transform(df_clean[numerical_cols])

    print(f"\nMissing Values Processing Summary:")
    print(f"Initial dataset size: {initial_rows:,} rows")
    print(f"Final clean dataset size: {len(df_clean):,} rows")
    print(f"Percentage of data retained: {(len(df_clean)/initial_rows)*100:.2f}%")

    print_subsection("2.2 Feature Engineering",
                    "Creating new features from existing data")
    
    df_clean['DateTime'] = pd.to_datetime(df_clean['Date'].astype(str) + ' ' + 
                                        df_clean['Time'].astype(str))
    
    df_clean['Hour'] = df_clean['DateTime'].dt.hour
    df_clean['DayOfWeek'] = df_clean['DateTime'].dt.dayofweek
    df_clean['Month'] = df_clean['DateTime'].dt.month
    df_clean['DayOfYear'] = df_clean['DateTime'].dt.dayofyear

    df_clean['NOx_NO2_ratio'] = df_clean['NOx(GT)'] / (df_clean['NO2(GT)'] + 1e-6)
    df_clean['CO_NMHC_ratio'] = df_clean['PT08.S1(CO)'] / (df_clean['NMHC(GT)'] + 1e-6)

    base_features = [
        'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)',
        'PT08.S5(O3)', 'T', 'RH', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'NMHC(GT)',
        'Hour', 'Month', 'DayOfYear', 'NOx_NO2_ratio', 'CO_NMHC_ratio'
    ]

    # Save feature names
    feature_names = pd.DataFrame({'feature_names': base_features})
    feature_names.to_csv('feature_names.csv', index=False)

    X = df_clean[base_features]
    y = df_clean['CO(GT)']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, base_features

# Step 3: Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print_section("STEP 3: MODEL TRAINING AND EVALUATION",
                 "Training and evaluating traditional machine learning models")

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print_subsection(f"3.1 {name} Model",
                        f"Training and evaluating {name} model")

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        print("\nCross-validation Results:")
        print("Fold\tR2 Score")
        print("-"*20)
        for fold, score in enumerate(cv_scores, 1):
            print(f"{fold}\t{score:.4f}")
        print(f"Mean\t{cv_scores.mean():.4f}")
        print(f"Std\t{cv_scores.std():.4f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\nTest Set Performance Metrics:")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'metrics': {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        }

    return results

# Step 4: Neural Network Implementation
def train_neural_network(X_train, X_test, y_train, y_test):
    print_section("STEP 4: NEURAL NETWORK IMPLEMENTATION",
                 "Building and training a deep learning model")

    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

    inputs = tf.keras.Input(shape=(X_train.shape[1],))
    x = Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.4)(x)

    residual = x
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([x, residual])

    residual = x
    x = Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([x, residual])

    x = Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        weight_decay=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train_tf, y_train_tf,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    y_pred = model.predict(X_test_tf, batch_size=32).flatten()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nPerformance Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return model, y_pred, history

def visualize_results(y_test, results, nn_pred, nn_history, selected_features):
    print_section("STEP 5: RESULTS VISUALIZATION",
                 "Creating visual representations of model performance")

    # Print model architecture and formulas
    print_subsection("MODEL ARCHITECTURE AND FORMULAS",
                    "Detailed explanation of model components and formulas")

    print("\n1. Linear Regression Model:")
    print("-"*80)
    print("Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε")
    print("Where:")
    print("  - y: Target variable (CO(GT))")
    print("  - β₀: Intercept")
    print("  - β₁ to βₙ: Coefficients for each feature")
    print("  - x₁ to xₙ: Input features")
    print("  - ε: Error term")
    print("\nOptimization: Minimize Mean Squared Error (MSE)")
    print("MSE = (1/n) * Σ(yᵢ - ŷᵢ)²")
    print("Where:")
    print("  - n: Number of samples")
    print("  - yᵢ: Actual value")
    print("  - ŷᵢ: Predicted value")

    print("\n2. Random Forest Model:")
    print("-"*80)
    print("Ensemble of Decision Trees:")
    print("Each tree prediction: ŷ = f(x₁, x₂, ..., xₙ)")
    print("Final prediction: ŷ = (1/T) * Σ ŷᵢ")
    print("Where:")
    print("  - T: Number of trees (n_estimators=100)")
    print("  - ŷᵢ: Prediction from i-th tree")
    print("\nFeature Importance Calculation:")
    print("Importance = (1/T) * Σ (node_impurity - left_impurity - right_impurity)")
    print("Where:")
    print("  - node_impurity: Gini impurity of the node")
    print("  - left_impurity: Gini impurity of left child")
    print("  - right_impurity: Gini impurity of right child")

    print("\n3. Neural Network Model:")
    print("-"*80)
    print("Architecture:")
    print("Input Layer (16 features) → Dense(256) → BatchNorm → Dropout(0.4)")
    print("→ Dense(128) → BatchNorm → Dropout(0.3) → Dense(256) → BatchNorm")
    print("→ Dense(64) → BatchNorm → Dropout(0.2) → Dense(256) → BatchNorm")
    print("→ Dense(32) → BatchNorm → Output(1)")
    
    print("\nActivation Functions:")
    print("ReLU: f(x) = max(0, x)")
    print("Linear: f(x) = x (output layer)")
    
    print("\nBatch Normalization:")
    print("x̂ = (x - μ) / √(σ² + ε)")
    print("y = γx̂ + β")
    print("Where:")
    print("  - μ: Mean of batch")
    print("  - σ²: Variance of batch")
    print("  - ε: Small constant")
    print("  - γ: Scale parameter")
    print("  - β: Shift parameter")
    
    print("\nDropout:")
    print("During training: x' = x * m")
    print("During inference: x' = x * p")
    print("Where:")
    print("  - m: Binary mask (Bernoulli distribution)")
    print("  - p: Dropout probability")

    print("\n4. Performance Metrics:")
    print("-"*80)
    print("Root Mean Squared Error (RMSE):")
    print("RMSE = √(MSE) = √[(1/n) * Σ(yᵢ - ŷᵢ)²]")
    
    print("\nMean Absolute Error (MAE):")
    print("MAE = (1/n) * Σ|yᵢ - ŷᵢ|")
    
    print("\nR² Score (Coefficient of Determination):")
    print("R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)")
    print("Where:")
    print("  - ȳ: Mean of actual values")

    print("\n5. Error Distribution Metrics:")
    print("-"*80)
    print("Skewness:")
    print("γ₁ = (1/n) * Σ[(xᵢ - μ)/σ]³")
    print("Where:")
    print("  - μ: Mean of errors")
    print("  - σ: Standard deviation of errors")
    
    print("\nKurtosis:")
    print("γ₂ = (1/n) * Σ[(xᵢ - μ)/σ]⁴ - 3")
    print("Where:")
    print("  - μ: Mean of errors")
    print("  - σ: Standard deviation of errors")

    # Create a directory for saving plots
    os.makedirs('plots', exist_ok=True)

    print_subsection("5.1 Model Performance Comparison",
                    "Comparing predictions across all models")

    # Calculate and print detailed statistics for each model
    print("\nDetailed Model Performance Statistics:")
    print("-"*80)
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R2':<10} {'Max Error':<12} {'Min Error':<12}")
    print("-"*80)

    for name, result in results.items():
        errors = y_test - result['y_pred']
        print(f"{name:<20} {result['metrics']['RMSE']:.4f} {result['metrics']['MAE']:.4f} "
              f"{result['metrics']['R2']:.4f} {errors.max():.4f} {errors.min():.4f}")

    # Neural Network statistics
    nn_errors = y_test - nn_pred
    nn_mse = mean_squared_error(y_test, nn_pred)
    nn_rmse = np.sqrt(nn_mse)
    nn_mae = mean_absolute_error(y_test, nn_pred)
    nn_r2 = r2_score(y_test, nn_pred)
    
    print(f"{'Neural Network':<20} {nn_rmse:.4f} {nn_mae:.4f} {nn_r2:.4f} "
          f"{nn_errors.max():.4f} {nn_errors.min():.4f}")
    print("-"*80)

    # Plot actual vs predicted for all models
    plt.figure(figsize=(15, 10))

    # Plot for traditional models
    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(2, 2, i)
        plt.scatter(y_test, result['y_pred'], alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{name} - Actual vs Predicted\nR2: {result["metrics"]["R2"]:.4f}')
        plt.grid(True)

    # Plot for Neural Network
    plt.subplot(2, 2, 3)
    plt.scatter(y_test, nn_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Neural Network - Actual vs Predicted\nR2: {nn_r2:.4f}')
    plt.grid(True)

    # Plot training history
    plt.subplot(2, 2, 4)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neural Network Training History\nFinal Loss: {:.4f}'.format(nn_history.history['loss'][-1]))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/model_performance.png')
    plt.close()

    print_subsection("5.2 Feature Importance Analysis",
                    "Analyzing feature importance in Random Forest model")

    # Plot feature importance for Random Forest
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 5 Most Important Features:")
    print("-"*50)
    for i, (feature, importance) in enumerate(zip(feature_importance['Feature'], feature_importance['Importance']), 1):
        print(f"{i}. {feature:<20} {importance:.4f}")
        if i == 5:
            break
    print("-"*50)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Random Forest)\nTotal Features: {}'.format(len(selected_features)))
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

    print_subsection("5.3 Error Distribution Analysis",
                    "Analyzing error distribution across models")

    # Calculate errors for each model
    errors = {}
    for name, result in results.items():
        errors[name] = y_test - result['y_pred']
    errors['Neural Network'] = y_test - nn_pred

    print("\nError Distribution Statistics:")
    print("-"*80)
    print(f"{'Model':<20} {'Mean Error':<12} {'Std Error':<12} {'Skewness':<12} {'Kurtosis':<12}")
    print("-"*80)

    for name, error in errors.items():
        print(f"{name:<20} {error.mean():.4f} {error.std():.4f} "
              f"{pd.Series(error).skew():.4f} {pd.Series(error).kurtosis():.4f}")
    print("-"*80)

    # Plot error distributions
    plt.figure(figsize=(15, 5))
    for i, (name, error) in enumerate(errors.items(), 1):
        plt.subplot(1, 3, i)
        sns.histplot(error, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'{name} Error Distribution\nMean: {error.mean():.4f}, Std: {error.std():.4f}')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/error_distribution.png')
    plt.close()

    print_subsection("5.4 Model Performance Metrics",
                    "Comparing model performance metrics")

    # Create a DataFrame of metrics
    metrics_data = []
    for name, result in results.items():
        metrics_data.append({
            'Model': name,
            'RMSE': result['metrics']['RMSE'],
            'MAE': result['metrics']['MAE'],
            'R2': result['metrics']['R2']
        })
    
    metrics_data.append({
        'Model': 'Neural Network',
        'RMSE': nn_rmse,
        'MAE': nn_mae,
        'R2': nn_r2
    })

    metrics_df = pd.DataFrame(metrics_data)
    
    print("\nDetailed Model Performance Metrics:")
    print("-"*80)
    print(metrics_df.to_string(index=False))
    print("-"*80)

    # Plot metrics
    plt.figure(figsize=(15, 5))
    metrics = ['RMSE', 'MAE', 'R2']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.barplot(x='Model', y=metric, data=metrics_df)
        plt.title(f'{metric} Comparison\nBest: {metrics_df[metric].min():.4f}')
        plt.xticks(rotation=45)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/metrics_comparison.png')
    plt.close()

    print("\nVisualization Summary:")
    print("="*80)
    print("1. Model Performance Comparison (model_performance.png):")
    print("   - Actual vs Predicted scatter plots for all models")
    print("   - Neural Network training history")
    print("   - R2 scores for each model")
    print("   - Formulas used for each model's predictions")
    
    print("\n2. Feature Importance Analysis (feature_importance.png):")
    print("   - Bar plot of feature importance from Random Forest")
    print("   - Top 5 most important features identified")
    print("   - Feature importance calculation formula")
    
    print("\n3. Error Distribution Analysis (error_distribution.png):")
    print("   - Histograms of prediction errors for each model")
    print("   - Error statistics including mean, std, skewness, and kurtosis")
    print("   - Error distribution formulas and calculations")
    
    print("\n4. Model Performance Metrics (metrics_comparison.png):")
    print("   - Comparison of RMSE, MAE, and R2 across all models")
    print("   - Best performing model for each metric")
    print("   - Detailed formulas for each metric")
    print("="*80)

    print("\nAll visualizations have been saved in the 'plots' directory.")
    print("You can find detailed performance metrics, formulas, and analysis in the plots.")

def main():
    print_section("AIR QUALITY INDEX PREDICTION PROJECT",
                 "Predicting CO levels using multiple machine learning models")

    df = load_and_explore_data()
    X_train, X_test, y_train, y_test, scaler, selected_features = preprocess_data(df)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    nn_model, nn_pred, nn_history = train_neural_network(X_train, X_test, y_train, y_test)
    
    # Add visualization step
    visualize_results(y_test, results, nn_pred, nn_history, selected_features)

    # Save models and scaler
    print_section("SAVING MODELS",
                 "Saving trained models and scaler for future predictions")

    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(results['Linear Regression']['model'], 'models/linear_model.joblib')
        joblib.dump(results['Random Forest']['model'], 'models/rf_model.joblib')
        nn_model.save('models/nn_model.keras')
        joblib.dump(scaler, 'models/scaler.joblib')
        print("\nModels and scaler saved successfully!")
    except Exception as e:
        print(f"\nError saving models: {e}")

    print_section("PROJECT COMPLETED",
                 "All models have been trained and evaluated successfully")

if __name__ == "__main__":
    main() 