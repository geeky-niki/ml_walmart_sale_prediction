import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pg_conn_comp import pg_connector  # Ensure correct import

# Function to fetch data from PostgreSQL
def fetch_data(schema_name, table_name):
    conn = pg_connector()  # Ensure you have a function to connect to the database
    if conn is None:
        raise Exception("Failed to connect to the database")
    
    query = f"SELECT * FROM {schema_name}.{table_name}"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Fetch the data from the base table in ati_tb schema
schema_name = 'ati_tb'
table_name = 'source_base_file_tbl'
data = fetch_data(schema_name, table_name)

# Convert Weekly_Sales to numeric, coercing errors
data['Weekly_Sales'] = pd.to_numeric(data['Weekly_Sales'], errors='coerce')

# Drop rows with NaN values in Weekly_Sales
data.dropna(subset=['Weekly_Sales'], inplace=True)

# Select features and target
features = ['IsHoliday', 'Dept', 'Temperature', 'Fuel_Price', 'MarkDown1', 
            'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 
            'Unemployment', 'Type', 'Size']
X = data[features]
y = data['Weekly_Sales']

# Handle categorical variables
X = pd.get_dummies(X, columns=['IsHoliday', 'Type'], drop_first=True)

# Save the feature names
feature_names = X.columns.tolist()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate each model
models = {
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(random_state=42, silent=True)
}

best_model = None
best_rmse = float('inf')

for name, model in models.items():
    print(f"Training and evaluating {name}...")
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, f'{name.lower()}_model.pkl')
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE for {name}: {rmse:.2f}')
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = (name, model)

# Best model results
best_name, best_model_instance = best_model

print(f"\nBest model: {best_name} with RMSE {best_rmse:.2f}\n")

# Example new data points for prediction (each day of the next week)
new_data = {
    'IsHoliday': [0, 0, 0, 0, 0, 0, 0],
    'Dept': [1, 1, 1, 1, 1, 1, 1],
    'Temperature': [65, 66, 67, 68, 69, 70, 71],
    'Fuel_Price': [3.2, 3.21, 3.22, 3.23, 3.24, 3.25, 3.26],
    'MarkDown1': [1000, 1000, 1000, 1000, 1000, 1000, 1000],
    'MarkDown2': [500, 500, 500, 500, 500, 500, 500],
    'MarkDown3': [100, 100, 100, 100, 100, 100, 100],
    'MarkDown4': [0, 0, 0, 0, 0, 0, 0],
    'MarkDown5': [50, 50, 50, 50, 50, 50, 50],
    'CPI': [211, 211, 211, 211, 211, 211, 211],
    'Unemployment': [6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5],
    'Type': ['A', 'A', 'A', 'A', 'A', 'A', 'A'],
    'Size': [150000, 150000, 150000, 150000, 150000, 150000, 150000]
}

# Convert to DataFrame
new_data_df = pd.DataFrame(new_data)

# Handle categorical variables
new_data_df = pd.get_dummies(new_data_df, columns=['IsHoliday', 'Type'], drop_first=True)

# Ensure the columns match the training data
for col in feature_names:
    if col not in new_data_df:
        new_data_df[col] = 0

# Reorder columns to match the training data
new_data_df = new_data_df[feature_names]

# Standardize the new data
new_data_df = scaler.transform(new_data_df)

# Load and predict with the best model
model = joblib.load(f'{best_name.lower()}_model.pkl')
predictions = model.predict(new_data_df)
print(f"Predicted sales for each day of the next week: {predictions}")