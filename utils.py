# -*- coding: utf-8 -*-
"""
Модуль utils.py
Описание: Вспомогательные функции для анализа данных в системе мониторинга вузов.
Содержит функции для проверки качества данных, визуализации признаков и анализа важности признаков.
Используется в приложении Streamlit для расширения функциональности.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# Настройка стиля графиков
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def check_data_quality(df):
    """
    Проверка качества данных: пропуски, статистика, типы данных
    
    Параметры:
    df (pandas.DataFrame): Входной датасет
    
    Возвращает:
    dict: Словарь с информацией о пропусках, статистике и типах данных
    """
    st.subheader("Анализ качества данных")
    
    # Проверка пропусков
    missing_values = df.isna().sum()
    missing_summary = pd.DataFrame({
        'Столбец': missing_values.index,
        'Количество пропусков': missing_values.values,
        'Процент пропусков': (missing_values / len(df) * 100).round(2)
    })
    st.write("Пропуски в данных:")
    st.dataframe(missing_summary)
    
    # Основная статистика
    stats = df.describe().round(2)
    st.write("Основная статистика по числовым столбцам:")
    st.dataframe(stats)
    
    # Типы данных
    dtypes_summary = pd.DataFrame({
        'Столбец': df.columns,
        'Тип данных': df.dtypes.values
    })
    st.write("Типы данных столбцов:")
    st.dataframe(dtypes_summary)
    
    result = {
        'missing': missing_summary,
        'stats': stats,
        'dtypes': dtypes_summary
    }
    return result

def plot_feature_distribution(df, feature, save_path=None):
    """
    Построение гистограммы распределения признака с ядерной оценкой плотности
    
    Параметры:
    df (pandas.DataFrame): Входной датасет
    feature (str): Название столбца для анализа
    save_path (str, optional): Путь для сохранения графика
    
    Возвращает:
    None: Отображает график в Streamlit
    """
    try:
        if feature not in df.columns:
            st.error(f"Признак '{feature}' отсутствует в данных")
            return
        if not np.issubdtype(df[feature].dtype, np.number):
            st.error("Признак должен быть числовым")
            return
        if df[feature].isna().all():
            st.error("Признак содержит только пропуски")
            return
        
        st.subheader(f"Распределение признака '{feature}'")
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax, color='skyblue')
        ax.set_title(f"Распределение признака '{feature}'")
        ax.set_xlabel(feature)
        ax.set_ylabel("Частота")
        st.pyplot(fig)
        
        if save_path:
            fig.savefig(save_path)
            st.write(f"График сохранён как {save_path}")
        
        plt.close(fig)
    except Exception as e:
        st.error(f"Ошибка при построении гистограммы: {e}")

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Построение графика важности признаков для модели регрессии
    
    Параметры:
    model: Обученная модель с атрибутом feature_importances_
    feature_names (list): Список названий признаков
    save_path (str, optional): Путь для сохранения графика
    
    Возвращает:
    None: Отображает график в Streamlit
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            st.error("Модель не поддерживает анализ важности признаков")
            return
        if len(feature_names) != len(model.feature_importances_):
            st.error("Количество признаков не соответствует модели")
            return
        
        st.subheader("Важность признаков модели")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x=model.feature_importances_, y=feature_names, ax=ax, palette='viridis')
        ax.set_title("Важность признаков в модели")
        ax.set_xlabel("Важность")
        ax.set_ylabel("Признак")
        st.pyplot(fig)
        
        if save_path:
            fig.savefig(save_path)
            st.write(f"График сохранён как {save_path}")
        
        plt.close(fig)
    except Exception as e:
        st.error(f"Ошибка при построении графика важности: {e}")
        
def save_dataframe(df, filename="data.csv"):
    """
    Сохранение датасета в файл
    
    Параметры:
    df (pandas.DataFrame): Датасет для сохранения
    filename (str): Имя файла
    
    Возвращает:
    bool: True, если сохранение успешно
    """
    try:
        df.to_csv(filename, index=False)
        st.write(f"Датасет сохранён как {filename}")
        return True
    except Exception as e:
        st.error(f"Ошибка при сохранении датасета: {e}")
        return False

def load_dataframe(filename="data.csv"):
    """
    Загрузка датасета из файла
    
    Параметры:
    filename (str): Имя файла
    
    Возвращает:
    pandas.DataFrame или None: Загруженный датасет или None при ошибке
    """
    try:
        if not os.path.exists(filename):
            st.error(f"Файл {filename} не найден")
            return None
        df = pd.read_csv(filename)
        st.write(f"Датасет загружен из {filename}")
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке датасета: {e}")
        return None