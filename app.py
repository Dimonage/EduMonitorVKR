import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os

from utils import (
    check_data_quality, plot_feature_distribution, plot_feature_importance,
    save_dataframe, load_dataframe, detect_outliers, summarize_features
)

# Настройка стиля визуализации
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Заголовок приложения
st.title("Система интеллектуального анализа данных мониторинга вузов EduMonitor")

# Инициализация состояния сессии
def initialize_session_state():
    session_keys = {
        'df_clean': None,
        'clusters': None,
        'kmeans': None,
        'model_ege': None,
        'model_niokr': None,
        'feature_names_ege': None,
        'feature_names_niokr': None,
        'data_loaded': False
    }
    for key, value in session_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value

 # 1. Загрузка данных
def load_data():
    uploaded_file = st.file_uploader("Загрузите датасет (.xlsx)", type=["xlsx"], key="file_uploader")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"Датасет загружен! Размер: {df.shape}")
            return df
        except Exception as e:
            st.error(f"Ошибка при загрузке данных: {e}")
            return None
    return None

# 2. Предобработка данных
def preprocess_data(df, target_col=None):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean = df_clean[numeric_cols]
    df_clean = df_clean.fillna(df_clean.mean())
    df_clean = df_clean.loc[:, df_clean.var() > 0]
    
    if target_col and target_col in df_clean.columns:
        if target_col == 'Общий объем научно-исследовательских и опытно-конструкторских работ (далее – НИОКР)':
            df_clean[target_col] = df_clean[target_col].clip(lower=1e-6)
        outliers = detect_outliers(df_clean, target_col)
        if outliers is not None:
            df_clean = df_clean[~outliers]
        df_clean = df_clean.dropna(subset=[target_col])
    
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    st.write(f"Данные обработаны. Размер после очистки: {df_clean.shape}")
    return df_clean

# 3. Отбор признаков
def select_features(X, y, k=20):
    try:
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        st.write(f"Отобрано {k} признаков: {selected_features}")
        return X[selected_features], selected_features
    except Exception as e:
        st.error(f"Ошибка при отборе признаков: {e}")
        return X, X.columns.tolist()