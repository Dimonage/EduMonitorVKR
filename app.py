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
