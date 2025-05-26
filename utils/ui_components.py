# utils/ui_components.py — визуальные компоненты Streamlit UI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


def render_data_exploration_ui(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("📭 Нет данных для анализа.")
        return

    if st.button("🔍 Проверить пропуски"):
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
        st.info(f"🔁 Найдено дубликатов: {count}")

    if st.button("🧹 Удалить дубликаты", key="clean_btn"):
        df_clean = df.drop_duplicates()
        st.session_state.df_clean = df_clean
        st.success(f"🧹 Дубликаты удалены. Новый размер: {df_clean.shape}")


def render_clustering_visuals(df: pd.DataFrame, clusters: np.ndarray):
    if df is None or clusters is None:
        st.warning("Нет данных для кластеризации")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("Кластер", errors="ignore")

    st.subheader("📌 Диаграмма рассеяния (Plotly)")
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox("X ось", numeric_cols, key="scatter_x")
    with col2:
        y = st.selectbox("Y ось", numeric_cols, key="scatter_y")

    if st.button("📈 Построить scatter plot"):
        df_plot = df.copy()
        df_plot["Кластер"] = clusters
        fig = px.scatter(
            df_plot, x=x, y=y, color="Кластер",
            title=f"Кластеры: {x} vs {y}", height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📦 BoxPlot")
    feat = st.selectbox("Признак для boxplot", numeric_cols, key="boxplot_feat")
    if st.button("📦 Построить boxplot"):
        df_plot = df.copy()
        df_plot["Кластер"] = clusters
        fig = px.box(
            df_plot, x="Кластер", y=feat, color="Кластер",
            title=f"BoxPlot по кластерам: {feat}", height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Гистограмма кластеров")
    if st.button("📉 Построить гистограмму кластеров"):
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index, y=cluster_counts.values,
            labels={"x": "Кластер", "y": "Количество"},
            title="Распределение по кластерам", height=500
        )
        st.plotly_chart(fig, use_container_width=True)
