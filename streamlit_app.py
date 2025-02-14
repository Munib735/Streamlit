import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
def load_data(file_path):
    df = pd.read_csv(file_path, skiprows=1)
    return df

# Split Data
def split_data(df, target_col1, target_col2, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col1, target_col2])
    y1 = df[target_col1]
    y2 = df[target_col2]
    X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=test_size, random_state=random_state)
    _, _, y2_train, y2_test = train_test_split(X, y2, test_size=test_size, random_state=random_state)
    return X_train, X_test, y1_train, y1_test, y2_train, y2_test

# Train Model
def train_model(X_train, y_train, param_grid):
    model = RandomForestRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2, y_pred

# SHAP Analysis
def shap_analysis(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    return shap_values

# Plot SHAP Summary
def plot_shap_summary(shap_values, X_train):
    shap.summary_plot(shap_values, X_train)

# Streamlit UI
def main():
    st.title("Gas Turbine Prediction Model")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("Data Preview:", df.head())
        
        target_col1 = st.selectbox("Select first target variable", df.columns)
        target_col2 = st.selectbox("Select second target variable", df.columns)
        
        if st.button("Train Model"):
            X_train, X_test, y1_train, y1_test, y2_train, y2_test = split_data(df, target_col1, target_col2)
            
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            
            st.write("Training Model for Response 1...")
            model1, best_params1 = train_model(X_train, y1_train, param_grid)
            rmse1, r2_1, y1_pred = evaluate_model(model1, X_test, y1_test)
            st.write(f"Response 1 - RMSE: {rmse1}, R2: {r2_1}")
            
            st.write("Training Model for Response 2...")
            model2, best_params2 = train_model(X_train, y2_train, param_grid)
            rmse2, r2_2, y2_pred = evaluate_model(model2, X_test, y2_test)
            st.write(f"Response 2 - RMSE: {rmse2}, R2: {r2_2}")
            
            if st.button("Perform SHAP Analysis"):
                shap_values1 = shap_analysis(model1, X_train)
                st.write("SHAP Summary for Response 1")
                plot_shap_summary(shap_values1, X_train)
                
                shap_values2 = shap_analysis(model2, X_train)
                st.write("SHAP Summary for Response 2")
                plot_shap_summary(shap_values2, X_train)

if __name__ == "__main__":
    main()
