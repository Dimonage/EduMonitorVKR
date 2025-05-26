# utils/ui_components.py — интерфейсные компоненты Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.io_tools import save_dataframe
from utils.analysis_tools import (
    show_missing_values,
    show_column_summary,
    check_duplicates
)
from utils.globals import saved_plots, plot_descriptions
from config import DATA_PATHS


def render_data_exploration_ui(df: pd.DataFrame):
    """Интерфейс для анализа данных"""
    if df is None or df.empty:
        st.warning("📭 Нет данных для анализа.")
        return

    if st.button("🔍 Проверить пропущенные значения"):
        show_missing_values(df)

    feature = st.selectbox("📌 Признак для гистограммы", df.select_dtypes(include=[np.number]).columns)

    if st.button("📊 Построить распределение", key="btn_dist"):
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax, color='skyblue')
        ax.set_title(f"Распределение: {feature}")
        st.pyplot(fig)
        fig.savefig("plots/distribution.png")
        saved_plots.append("plots/distribution.png")
        plot_descriptions.append(f"Распределение признака {feature}")
        plt.close(fig)

    if st.button("📋 Характеристики признаков", key="btn_summary"):
        show_column_summary(df)

    if st.button("🔁 Проверить дубликаты", key="btn_dup"):
        check_duplicates(df)

    if st.button("🧹 Удалить дубликаты", key="btn_drop_dup"):
        df_clean = df.drop_duplicates()
        st.session_state.df_clean = df_clean
        st.success(f"✅ Дубликаты удалены. Размер: {df_clean.shape}")
        save_dataframe(df_clean, DATA_PATHS["processed"])


def render_clustering_visuals(df: pd.DataFrame, clusters: np.ndarray):
    """Визуализация кластеризации"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("Кластер", errors="ignore")

    st.subheader("📈 Диаграмма рассеяния")
    x = st.selectbox("Ось X", numeric_cols, key="scatter_x")
    y = st.selectbox("Ось Y", numeric_cols, key="scatter_y")
    if st.button("📍 Построить scatter plot"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x, y=y, hue=clusters, palette="deep", ax=ax, s=100)
        ax.set_title(f"Кластеры: {x} vs {y}")
        st.pyplot(fig)
        fig.savefig("plots/cluster_scatter.png")
        saved_plots.append("plots/cluster_scatter.png")
        plot_descriptions.append(f"Scatter plot: {x} vs {y}")
        plt.close(fig)

    st.subheader("🎯 Box plot по кластерам")
    feat = st.selectbox("Признак", numeric_cols, key="boxplot_feat")
    if st.button("📦 Построить box plot"):
        df_plot = df.copy()
        df_plot["Кластер"] = clusters
        fig, ax = plt.subplots()
        sns.boxplot(x="Кластер", y=feat, data=df_plot, ax=ax, palette="Set2")
        st.pyplot(fig)
        fig.savefig("plots/cluster_boxplot.png")
        saved_plots.append("plots/cluster_boxplot.png")
        plot_descriptions.append(f"Box plot по кластерам для {feat}")
        plt.close(fig)

    st.subheader("📊 Гистограмма кластеров")
    if st.button("📉 Построить гистограмму кластеров"):
        fig, ax = plt.subplots()
        sns.countplot(x=clusters, palette="Set1", ax=ax)
        ax.set_title("Распределение по кластерам")
        st.pyplot(fig)
        fig.savefig("plots/cluster_histogram.png")
        saved_plots.append("plots/cluster_histogram.png")
        plot_descriptions.append("Гистограмма кластеров")
        plt.close(fig)
