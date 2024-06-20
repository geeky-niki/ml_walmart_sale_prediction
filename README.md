# 📈 Weekly Sales Prediction

## Overview
This project aims to predict the weekly sales of a retail store using machine learning models. By analyzing historical sales data, we can build a model to forecast future sales, aiding in business decision-making.

## Project Steps
1. **🗄️ Data Fetching:** 
   - Retrieve historical sales data from a PostgreSQL database.

2. **🧹 Data Preprocessing:** 
   - Clean the data by handling missing values and converting necessary columns to numerical types.

3. **🔧 Feature Engineering:** 
   - Select relevant features for the model, such as `IsHoliday`, `Dept`, `Temperature`, `Fuel_Price`, etc.
   - Convert categorical variables to dummy variables for model compatibility.

4. **🤖 Model Training and Evaluation:** 
   - Train multiple machine learning models (XGBoost, LightGBM, CatBoost) on the processed data.
   - Evaluate models using Root Mean Squared Error (RMSE) to identify the best performing model.

5. **🔮 Making Predictions:** 
   - Use the best model to make sales predictions on new data points.

## Usage
1. **📥 Fetch Data:**
   - Connect to the PostgreSQL database and fetch data using the `fetch_data` function.

2. **🧼 Preprocess Data:**
   - Convert `Weekly_Sales` to numeric, handle missing values, and create dummy variables for categorical features.

3. **🏋️‍♂️ Train Models:**
   - Train XGBoost, LightGBM, and CatBoost models on the training data.
   - Evaluate each model's performance and save the best one.

4. **🔍 Predict Sales:**
   - Load the best model and make predictions on new data points.

## Conclusion
This project demonstrates the practical application of machine learning to predict retail sales. By leveraging different models and evaluating their performance, we can select the best model for accurate forecasts. This predictive capability can significantly enhance business planning and operational efficiency.
