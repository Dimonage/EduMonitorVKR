import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from utils import io_tools
import numpy as np


def plot_histogram(df, feature):
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax, color='skyblue')
    ax.set_title(f"Распределение: {feature}")
    st.pyplot(fig)
    plt.close(fig)


def render_data_exploration_ui(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("📭 Нет данных для отображения.")
        return

    if st.button("🔎 Проверить качество данных"):
        missing = df.isna().sum()
        st.dataframe(pd.DataFrame({
            "Столбец": missing.index,
            "Пропусков": missing.values,
            "%": (missing / len(df) * 100).round(2)
        }))

    feature = st.selectbox("Выберите признак для гистограммы", df.select_dtypes(include=[np.number]).columns)
    if st.button("📊 Построить распределение", key="dist_btn"):
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax, color='skyblue')
        ax.set_title(f"Распределение: {feature}")
        st.pyplot(fig)
        plt.close(fig)

    if st.button("📋 Характеристики признаков", key="summary_btn"):
        summary = pd.DataFrame({
            "Признак": df.columns,
            "Тип": df.dtypes,
            "Среднее": df.mean(numeric_only=True),
            "Дисперсия": df.var(numeric_only=True),
            "Уникальные": df.nunique()
        })
        st.dataframe(summary)

    if st.button("🔁 Проверить дубликаты", key="dup_btn"):
        count = df.duplicated().sum()
        st.info(f"Найдено дубликатов: {count}")

    if st.button("🧹 Удалить дубликаты", key="clean_btn"):
        df_clean = df.drop_duplicates()
        st.session_state.df_clean = df_clean
        st.success(f"Дубликаты удалены. Новый размер: {df_clean.shape}")
        io_tools.save_dataframe(df_clean, "data/processed_data.csv")


def render_clustering_visuals(df, clusters):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("Кластер", errors='ignore')

    st.subheader("Диаграмма рассеяния")
    x = st.selectbox("X ось", numeric_cols, key="scatter_x")
    y = st.selectbox("Y ось", numeric_cols, key="scatter_y")
    if st.button("📈 Построить scatter plot"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x, y=y, hue=clusters, palette="deep", s=100, ax=ax)
        ax.set_title(f"Кластеры: {x} vs {y}")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("📦 Box plot")
    feat = st.selectbox("Признак", numeric_cols, key="boxplot")
    if st.button("🎯 Построить box plot"):
        df_plot = df.copy()
        df_plot['Кластер'] = clusters
        fig, ax = plt.subplots()
        sns.boxplot(x='Кластер', y=feat, data=df_plot, palette="deep", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("📊 Гистограмма кластеров")
    if st.button("📉 Построить гистограмму"):
        fig, ax = plt.subplots()
        sns.countplot(x=clusters, palette="deep", ax=ax)
        ax.set_title("Распределение по кластерам")
        st.pyplot(fig)
        plt.close(fig)