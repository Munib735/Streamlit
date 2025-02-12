import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error as mse
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from hyperopt import hp, tpe, Trials, fmin
import shap
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    uploaded_file = st.file_uploader("Coal data", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv("Coal data")
        st.success("File uploaded successfully!")
        return df
    else:
        st.warning("Please upload a CSV file to proceed.")
        return None

def main():
    st.title("Steam Turbine Data Analysis")
    df = load_data()
    if df is not None:
        st.write("Data Preview:")
        st.write(df)
        # Proceed with further analysis or modeling using df

if __name__ == "__main__":
    main()

# Function to normalize features
def normalize_features(df, features):
    df_normalized = df.copy()
    df_normalized[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
    return df_normalized

# Function to plot KDE
def plot_kde(df_normalized, features):
    plt.figure(figsize=(5, 3.65))
    for feature in features:
        sns.kdeplot(df_normalized[feature], label=feature)
    plt.xlabel('Feature', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tick_params(which='both', direction='out', left=True, bottom=True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55), ncol=4, frameon=True, fontsize=10)
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1)
    plt.tick_params(which='both', direction='out', left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(plt)

# Function to split data
def split_data(df, target_column):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

# Function to perform hyperparameter optimization
def optimize_hyperparameters(x_train, y_train, x_test, y_test):
    space = {
        'eta': hp.uniform('eta', 0.01, 0.3),
        'gamma': hp.uniform('gamma', 0.2, 0.5),
        'reg_lambda': hp.uniform('reg_lambda', 0.1, 1.5),
        'max_depth': hp.quniform('max_depth', 4, 10, 1),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 50)
    }

    def objective(params):
        params['max_depth'] = int(params['max_depth'])
        model = XGBRegressor(
            eta=params['eta'],
            gamma=params['gamma'],
            reg_lambda=params['reg_lambda'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            n_estimators=int(params['n_estimators']),
            eval_metric='rmse',
            seed=42
        )
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        y_pred = model.predict(x_test)
        rmse_test = np.sqrt(mse(y_test, y_pred))
        return {'loss': rmse_test, 'status': 'ok', 'params': params}

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(30)
    )

    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    return best_params, trials

# Function to train and evaluate the model
def train_and_evaluate(x_train, y_train, x_test, y_test, best_params):
    model = XGBRegressor(**best_params, verbose=False, seed=50)
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = sqrt(mse(y_train, y_pred_train))
    rmse_test = sqrt(mse(y_test, y_pred_test))
    return model, r2_train, r2_test, rmse_train, rmse_test, y_pred_train, y_pred_test

# Function to plot predictions
def plot_predictions(y_train, y_pred_train, y_test, y_pred_test):
    plt.figure(figsize=(8, 6))
    sns.set_style("white")
    g = sns.JointGrid(x=y_test, y=y_pred_test, height=4)
    g.plot_joint(sns.scatterplot, color='blue', alpha=0.8, label='Test')
    sns.scatterplot(x=y_train, y=y_pred_train, color='orange', ax=g.ax_joint, alpha=0.8, label='Train')
    sns.regplot(x=y_test, y=y_pred_test, ax=g.ax_joint, scatter=False)
    g.ax_joint.legend()
    sns.histplot(x=y_train, ax=g.ax_marg_x, color='orange', kde=True)
    sns.histplot(y=y_test, ax=g.ax_marg_y, color='blue', kde=True)
    g.set_axis_labels("Actual CCGPP Efficiency (%)", "Predicted CCGPP Efficiency (%)", fontsize=10)
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot SHAP values
def plot_shap_values(model, x):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(x)
    shap.summary_plot(shap_values, x)
    plt.tight_layout()
    st.pyplot(plt)
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
