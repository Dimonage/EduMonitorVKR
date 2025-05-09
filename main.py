# -*- coding: utf-8 -*-

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
import utils

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
        outliers = utils.detect_outliers(df_clean, target_col)
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
        
        # График рассеяния
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Фактические значения")
        ax.set_ylabel("Предсказанные значения")
        ax.set_title(f"Сравнение фактических и предсказанных значений ({target_name})")
        st.pyplot(fig)
        
        # Сохранение графика
        save_path = f"plots/regression_scatter_{target_name.replace(' ', '_')}_{len(utils.saved_plots)}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        utils.saved_plots.append(save_path)
        # Добавление читаемого описания
        description = f"Сравнение фактических и предсказанных значений: {target_name}"
        utils.plot_descriptions.append(description)
        st.write(f"График сохранён как {save_path}")
        
        plt.close(fig)
        
        # График важности признаков
        utils.plot_feature_importance(model, feature_names)
        
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
        
        # Сокращение имен признаков для имени файла, так как при изначальной длинны выдавало ошибку
        max_length = 50  
        safe_feature_x = feature_x.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')[:max_length]
        safe_feature_y = feature_y.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')[:max_length]
        
        # Формирование безопасного имени файла
        save_path = f"plots/cluster_scatter_{safe_feature_x}_vs_{safe_feature_y}_{len(utils.saved_plots)}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        utils.saved_plots.append(save_path)
        # Добавление читаемого описания, так как изначально текст был с прочерками
        description = f"Диаграмма рассеяния кластеров: {feature_x} против {feature_y}"
        utils.plot_descriptions.append(description)
        st.write(f"График сохранён как {save_path}")
        
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
        
        # Сохранение графика
        save_path = f"plots/cluster_distribution_{len(utils.saved_plots)}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        utils.saved_plots.append(save_path)
        # Добавление читаемого описания
        description = "Распределение вузов по кластерам"
        utils.plot_descriptions.append(description)
        st.write(f"График сохранён как {save_path}")
        
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
        
        # Сохранение графика
        save_path = f"plots/cluster_boxplot_{feature.replace(' ', '_')}_{len(utils.saved_plots)}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        utils.saved_plots.append(save_path)
        # Добавление читаемого описания
        description = f"Распределение признака по кластерам: {feature}"
        utils.plot_descriptions.append(description)
        st.write(f"График сохранён как {save_path}")
        
        plt.close(fig)
    except Exception as e:
        st.error(f"Ошибка при построении box plot: {e}")

# 11. Построение тепловой карты корреляций
def plot_correlation_heatmap(df):
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df.corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Тепловая карта корреляций")
        st.pyplot(fig)
        
        # Сохранение графика
        save_path = f"plots/correlation_heatmap_{len(utils.saved_plots)}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        utils.saved_plots.append(save_path)
        # Добавление читаемого описания
        description = "Тепловая карта корреляций между признаками"
        utils.plot_descriptions.append(description)
        st.write(f"График сохранён как {save_path}")
        
        plt.close(fig)
    except Exception as e:
        st.error(f"Ошибка при построении тепловой карты: {e}")

# 12. Сохранение модели
def save_model(model, filename="model.pkl"):
    try:
        joblib.dump(model, filename)
        st.write(f"Модель сохранена как {filename}")
    except Exception as e:
        st.error(f"Ошибка при сохранении модели: {e}")

# 13. Загрузка модели
def load_model(filename="model.pkl"):
    try:
        model = joblib.load(filename)
        st.write(f"Модель загружена из {filename}")
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

# Основной интерфейс
def main():
    initialize_session_state()
    
    st.subheader("Загрузка данных")
    if not st.session_state.data_loaded:
        df = load_data()
        if df is not None:
            st.session_state.df_clean = preprocess_data(df)
            if st.session_state.df_clean is not None and not st.session_state.df_clean.empty:
                utils.save_dataframe(st.session_state.df_clean, "processed_data.csv")
                st.session_state.data_loaded = True
                st.experimental_rerun()
            else:
                st.error("Ошибка: обработанный датасет пуст или некорректен")
    else:
        st.success(f"Датасет уже загружен! Размер: {st.session_state.df_clean.shape}")
    
    if st.session_state.df_clean is None or not st.session_state.data_loaded:
        st.warning("Пожалуйста, загрузите данные для продолжения.")
        return
    
    st.subheader("Выберите задачу")
    task = st.selectbox("Задача", ["Анализ данных", "Предсказание ЕГЭ", "Предсказание НИОКР", "Кластеризация", "Тепловая карта"], key="task")
    
    if task == "Анализ данных":
        st.subheader("Анализ данных")
        if st.button("Проверить качество данных"):
            utils.check_data_quality(st.session_state.df_clean)
        
        numeric_cols = [
            col for col in st.session_state.df_clean.select_dtypes(include=[np.number]).columns
            if st.session_state.df_clean[col].var() > 0 and not st.session_state.df_clean[col].isna().all()
        ]
        feature = st.selectbox("Выберите признак для анализа распределения", numeric_cols, key="dist_feature")
        if st.button("Показать распределение признака"):
            utils.plot_feature_distribution(st.session_state.df_clean, feature)
        
        if st.button("Суммаризировать признаки"):
            utils.summarize_features(st.session_state.df_clean)
    
        if st.button("Проверить дубликаты"):
            utils.check_duplicates(st.session_state.df_clean)

        if st.button("Удалить дубликаты"):
            st.session_state.df_clean = utils.remove_duplicates(st.session_state.df_clean)
            utils.save_dataframe(st.session_state.df_clean, "processed_data.csv")

        st.subheader("Анализ корреляций")
        corr_cols = st.multiselect("Выберите столбцы для корреляции", st.session_state.df_clean.columns, key="corr_cols")
        if st.button("Показать корреляции"):
            utils.calculate_selected_correlations(st.session_state.df_clean, corr_cols)

        if st.button("Экспортировать отфильтрованные данные"):
            utils.export_filtered_data(st.session_state.df_clean)
            with open("filtered_data_export.csv", "rb") as file:
                st.download_button(
                    label="Скачать отфильтрованный датасет",
                    data=file,
                    file_name="filtered_data_export.csv",
                    mime="text/csv"
                )

        st.subheader("Фильтрация данных")
        filter_col = st.selectbox("Выберите столбец для фильтрации", st.session_state.df_clean.columns, key="filter_col")
        filter_value = st.text_input("Введите значение для фильтрации", key="filter_value")
        if st.button("Применить фильтр"):
            filtered_df = utils.filter_data(st.session_state.df_clean, filter_col, filter_value)
            st.session_state.df_clean = filtered_df
            utils.save_dataframe(st.session_state.df_clean, "filtered_data.csv")

    elif task == "Предсказание ЕГЭ":
        st.subheader("Предсказание среднего балла ЕГЭ")
        target_col = 'Средний балл ЕГЭ студентов, принятых по результатам ЕГЭ на обучение по очной форме по программам бакалавриата и специалитета за счет средств соответствующих бюджетов бюджетной системы РФ'
        if target_col in st.session_state.df_clean.columns:
            if st.button("Обучить модель ЕГЭ"):
                X_train, X_test, y_train, y_test, scaler, feature_cols = split_data(
                    st.session_state.df_clean, target_col, log_transform=False
                )
                st.session_state.model_ege = train_regression_model(X_train, y_train)
                st.session_state.feature_names_ege = feature_cols
                rmse, mae = evaluate_regression_model(
                    st.session_state.model_ege, X_test, y_test, "Средний балл ЕГЭ",
                    st.session_state.feature_names_ege
                )
                if st.session_state.model_ege:
                    save_model(st.session_state.model_ege, "model_ege.pkl")
        else:
            st.error("Целевая переменная для ЕГЭ отсутствует!")
    
    elif task == "Предсказание НИОКР":
        st.subheader("Предсказание объема НИОКР")
        target_col = 'Общий объем научно-исследовательских и опытно-конструкторских работ (далее – НИОКР)'
        if target_col in st.session_state.df_clean.columns:
            if st.button("Обучить модель НИОКР"):
                df_niokr = preprocess_data(st.session_state.df_clean, target_col)
                X_train, X_test, y_train, y_test, scaler, feature_cols = split_data(
                    df_niokr, target_col, log_transform=True
                )
                st.session_state.model_niokr = train_regression_model(X_train, y_train)
                st.session_state.feature_names_niokr = feature_cols
                rmse, mae = evaluate_regression_model(
                    st.session_state.model_niokr, X_test, y_test, "Объем НИОКР",
                    st.session_state.feature_names_niokr, log_transform=True
                )
                if st.session_state.model_niokr:
                    save_model(st.session_state.model_niokr, "model_niokr.pkl")
        else:
            st.error("Целевая переменная для НИОКР отсутствует!")
    
    elif task == "Кластеризация":
        st.subheader("Кластеризация вузов")
        if st.button("Выполнить кластеризацию"):
            st.session_state.clusters, st.session_state.kmeans, cluster_scaler = cluster_vuz(
                st.session_state.df_clean, n_clusters=3
            )
            if st.session_state.clusters is not None:
                st.session_state.df_clean['Кластер'] = st.session_state.clusters
                save_model(st.session_state.kmeans, "model_kmeans.pkl")
                utils.save_dataframe(st.session_state.df_clean, "clustered_data.csv")
        
        if st.session_state.clusters is not None:
            numeric_cols = [
                col for col in st.session_state.df_clean.select_dtypes(include=[np.number]).columns
                if st.session_state.df_clean[col].var() > 0 and not st.session_state.df_clean[col].isna().all() and col != 'Кластер'
            ]
            
            st.subheader("Диаграмма рассеяния кластеров")
            feature_x = st.selectbox("Ось X", numeric_cols, key="scatter_x")
            feature_y = st.selectbox("Ось Y", numeric_cols, key="scatter_y")
            if st.button("Показать диаграмму рассеяния"):
                plot_cluster_scatter(st.session_state.df_clean, st.session_state.clusters, feature_x, feature_y)
            
            st.subheader("Гистограмма распределения кластеров")
            if st.button("Показать гистограмму"):
                plot_cluster_distribution(st.session_state.clusters)
            
            st.subheader("Box Plot по кластерам")
            feature_box = st.selectbox("Признак для Box Plot", numeric_cols, key="boxplot")
            if st.button("Показать Box Plot"):
                plot_cluster_boxplot(st.session_state.df_clean, st.session_state.clusters, feature_box)
        else:
            st.error("Сначала выполните кластеризацию!")
    
    elif task == "Тепловая карта":
        st.subheader("Тепловая карта корреляций")
        if st.button("Показать тепловую карту"):
            plot_correlation_heatmap(st.session_state.df_clean)
    
    st.subheader("Управление моделями")
    if st.button("Сохранить все модели"):
        if st.session_state.model_ege:
            save_model(st.session_state.model_ege, "model_ege.pkl")
        if st.session_state.model_niokr:
            save_model(st.session_state.model_niokr, "model_niokr.pkl")
        if st.session_state.kmeans:
            save_model(st.session_state.kmeans, "model_kmeans.pkl")
        if not st.session_state.model_ege and not st.session_state.model_niokr and not st.session_state.kmeans:
            st.warning("Нет моделей для сохранения!")
    
    if st.button("Загрузить модели"):
        st.session_state.model_ege = load_model("model_ege.pkl")
        st.session_state.model_niokr = load_model("model_niokr.pkl")
        st.session_state.kmeans = load_model("model_kmeans.pkl")
    
    st.subheader("Экспорт отчёта")
    standard_text = st.text_area(
        "Введите стандартизированный текст для отчёта (или оставьте пустым для текста по умолчанию)",
        height=200,
        key="standard_text"
    )
    if st.button("Экспортировать отчёт в Word"):
        if utils.saved_plots:
            with st.spinner("Создание отчёта..."):
                export_success = utils.export_to_word(
                    output_file="edu_monitor_report.docx",
                    standard_text=standard_text if standard_text.strip() else None
                )
                if export_success:
                    with open("edu_monitor_report.docx", "rb") as file:
                        st.download_button(
                            label="Скачать отчёт",
                            data=file,
                            file_name="edu_monitor_report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
        else:
            st.warning("Нет графиков для экспорта! Выполните анализ данных или визуализации.")
    
    if st.button("Очистить сохранённые графики"):
        utils.clear_saved_plots()

if __name__ == "__main__":
    main()