# utils/analysis_tools.py — инструменты для анализа качества данных

import pandas as pd
import streamlit as st
from scipy import stats


def detect_outliers_iqr(df: pd.DataFrame, column: str, iqr_factor: float = 1.5) -> pd.Series:
    """Выявление выбросов по IQR"""
    try:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        return (df[column] < q1 - iqr_factor * iqr) | (df[column] > q3 + iqr_factor * iqr)
    except Exception as e:
        st.error(f"Ошибка при определении выбросов: {e}")
        return pd.Series([False] * len(df))


def show_missing_values(df: pd.DataFrame):
    """Отображение таблицы пропусков"""
    missing = df.isna().sum()
    percent = (missing / len(df) * 100).round(2)
    st.subheader("📉 Пропущенные значения")
    st.dataframe(pd.DataFrame({
        "Столбец": missing.index,
        "Пропусков": missing.values,
        "%": percent.values
    }))


def show_column_summary(df: pd.DataFrame):
    """Показ базовой статистики по признакам"""
    st.subheader("📊 Характеристики признаков")
    summary = pd.DataFrame({
        "Признак": df.columns,
        "Тип": df.dtypes,
        "Среднее": df.mean(numeric_only=True),
        "Дисперсия": df.var(numeric_only=True),
        "Уникальные": df.nunique()
    })
    st.dataframe(summary)


def check_duplicates(df: pd.DataFrame):
    """Проверка и сообщение о количестве дубликатов"""
    count = df.duplicated().sum()
    st.info(f"🔁 Найдено дубликатов: {count}")
    return count
