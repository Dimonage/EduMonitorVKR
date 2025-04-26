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
    
# 4. Разделение данных для регрессии
def split_data(df, target_col, test_size=0.2, random_state=42, log_transform=False):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if log_transform:
        y = np.log1p(y)
        st.write("Применено логарифмирование к целевой переменной")
    
    X, selected_features = select_features(X, y, k=20)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    
    st.write(f"Данные разделены. Размер тренировочной выборки: {X_train.shape}")
    st.write(f"Размер тестовой выборки: {X_test.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, selected_features

# 5. Обучение модели регрессии
def train_regression_model(X_train, y_train, model_type="rf"):
    if model_type == "rf":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        st.write("Лучшие гиперпараметры:", grid_search.best_params_)
    else:
        raise ValueError("Неизвестный тип модели")
    
    st.write("Модель регрессии обучена.")
    return model

# 6. Оценка модели регрессии
def evaluate_regression_model(model, X_test, y_test, target_name, feature_names, log_transform=False):
    try:
        y_pred = model.predict(X_test)
        
        if log_transform:
            y_pred = np.expm1(y_pred)
            y_test = np.expm1(y_test)
        
        y_pred = np.clip(y_pred, 0, 1e10)
        y_test = np.clip(y_test, 0, 1e10)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        st.write(f"RMSE для {target_name}: {rmse:.2f}")
        st.write(f"MAE для {target_name}: {mae:.2f}")
        
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Фактические значения")
        ax.set_ylabel("Предсказанные значения")
        ax.set_title(f"Сравнение фактических и предсказанных значений ({target_name})")
        st.pyplot(fig)
        plt.close(fig)
        
        plot_feature_importance(model, feature_names, save_path=f"{target_name.lower().replace(' ', '_')}_importance.png")
        
        return rmse, mae
    except Exception as e:
        st.error(f"Ошибка при оценке модели: {e}")
        return None, None
    
# 7. Кластеризация вузов
def cluster_vuz(df, n_clusters=3):
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        st.write(f"Кластеризация выполнена. Количество кластеров: {n_clusters}")
        return clusters, kmeans, scaler
    except Exception as e:
        st.error(f"Ошибка при кластеризации: {e}")
        return None, None, None
    
# 8. Визуализация кластеров: Диаграмма рассеяния
def plot_cluster_scatter(df, clusters, feature_x, feature_y):
    try:
        if clusters is None:
            st.error("Кластеризация не выполнена")
            return
        if feature_x not in df.columns or feature_y not in df.columns:
            st.error(f"Признаки '{feature_x}' или '{feature_y}' отсутствуют")
            return
        if df[feature_x].isna().any() or df[feature_y].isna().any():
            st.error("Выбранные признаки содержат NaN")
            return
        if not np.issubdtype(df[feature_x].dtype, np.number) or not np.issubdtype(df[feature_y].dtype, np.number):
            st.error("Выбранные признаки должны быть числовыми")
            return
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feature_x, y=feature_y, hue=clusters, palette="deep", s=100, ax=ax)
        ax.set_title(f"Кластеризация вузов: {feature_x} vs {feature_y}")
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.legend(title="Кластер")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Ошибка при построении диаграммы рассеяния: {e}")

# 9. Визуализация кластеров: Гистограмма распределения
def plot_cluster_distribution(clusters):
    try:
        if clusters is None:
            st.error("Кластеризация не выполнена")
            return
        fig, ax = plt.subplots()
        sns.countplot(x=clusters, palette="deep", ax=ax)
        ax.set_title("Распределение вузов по кластерам")
        ax.set_xlabel("Кластер")
        ax.set_ylabel("Количество вузов")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Ошибка при построении гистограммы: {e}")

# 10. Визуализация кластеров: Box Plot
def plot_cluster_boxplot(df, clusters, feature):
    try:
        if clusters is None:
            st.error("Кластеризация не выполнена")
            return
        if feature not in df.columns:
            st.error(f"Признак '{feature}' отсутствует")
            return
        if df[feature].isna().any():
            st.error("Признак содержит NaN")
            return
        if not np.issubdtype(df[feature].dtype, np.number):
            st.error("Признак должен быть числовым")
            return
        
        df_plot = df.copy()
        df_plot['Кластер'] = clusters
        fig, ax = plt.subplots()
        sns.boxplot(x='Кластер', y=feature, data=df_plot, palette="deep", ax=ax)
        ax.set_title(f"Распределение признака '{feature}' по кластерам")
        ax.set_xlabel("Кластер")
        ax.set_ylabel(feature)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Ошибка при построении box plot: {e}")
