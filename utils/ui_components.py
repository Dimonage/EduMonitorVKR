# ui_components.py — визуализация анализа и кластеризации

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.io_tools import save_dataframe
from utils.analysis_tools import (
    show_missing_values,
    show_column_summary,
    check_duplicates
)
from config import DATA_PATHS


def render_data_exploration_ui(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("📭 Нет данных для анализа.")
        return

    if st.button("🔍 Проверить пропущенные значения"):
        show_missing_values(df)

    feature = st.selectbox("📌 Признак для гистограммы", df.select_dtypes(include=[np.number]).columns)
    if st.button("📊 Построить распределение", key="btn_dist"):
        fig = px.histogram(df, x=feature, nbins=30, title=f"Распределение: {feature}")
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title="Частота",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

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
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("Кластер", errors="ignore")

    st.subheader("📈 Диаграмма рассеяния")
    x = st.selectbox("Ось X", numeric_cols, key="scatter_x")
    y = st.selectbox("Ось Y", numeric_cols, key="scatter_y")
    if st.button("📍 Построить scatter plot"):
        plot_cluster_scatter(df, clusters, x, y)

    st.subheader("🎯 Box plot по кластерам")
    feat = st.selectbox("Признак", numeric_cols, key="boxplot_feat")
    if st.button("📦 Построить box plot"):
        plot_cluster_boxplot(df, clusters, feat)

    st.subheader("📊 Гистограмма кластеров")
    if st.button("📉 Построить гистограмму кластеров"):
        df_plot = pd.DataFrame({"Кластер": clusters})
        fig = px.histogram(
            df_plot,
            x="Кластер",
            color="Кластер",
            title="Распределение по кластерам",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_cluster_scatter(df: pd.DataFrame, clusters: np.ndarray, x: str, y: str):
    df_plot = df.copy()
    df_plot["Кластер"] = clusters
    fig = px.scatter(
        df_plot,
        x=x,
        y=y,
        color="Кластер",
        title=f"Кластеры: {x} vs {y}",
        height=600,
        hover_data=df_plot.columns
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_cluster_boxplot(df: pd.DataFrame, clusters: np.ndarray, feature: str):
    df_plot = df.copy()
    df_plot["Кластер"] = clusters
    fig = px.box(
        df_plot,
        x="Кластер",
        y=feature,
        color="Кластер",
        title=f"Boxplot по кластерам: {feature}",
        height=600,
        points="outliers"
    )
    st.plotly_chart(fig, use_container_width=True)
