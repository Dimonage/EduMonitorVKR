# visualization.py — визуализация метрик и корреляций

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.decorators import timeit, handle_errors


class Visualizer:

    @handle_errors
    @timeit
    def plot_prediction_scatter(self, y_true, y_pred, label: str):
        df = pd.DataFrame({
            "Фактическое значение": y_true,
            "Предсказание": y_pred
        })
        fig = px.scatter(
            df,
            x="Фактическое значение",
            y="Предсказание",
            title=f"Сравнение предсказания и факта: {label}",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    @handle_errors
    @timeit
    def plot_feature_importance(self, model, feature_names: list[str]):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        features = [feature_names[i] for i in sorted_idx[:10]]
        scores = importances[sorted_idx[:10]]

        df = pd.DataFrame({"Признак": features, "Важность": scores})
        fig = px.bar(
            df,
            x="Важность",
            y="Признак",
            orientation="h",
            title="Топ-10 важных признаков модели",
            color="Важность",
            color_continuous_scale="viridis",
            height=500
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    @handle_errors
    @timeit
    def plot_correlation_heatmap(self, df: pd.DataFrame, top_k: int = 30):
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] > top_k:
            top_features = numeric_df.corr().abs().mean().sort_values(ascending=False).head(top_k).index
            corr = numeric_df[top_features].corr()
        else:
            corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Интерактивная корреляционная матрица",
            labels=dict(x="Признаки", y="Признаки", color="Корреляция")
        )
        fig.update_layout(
            width=900,
            height=900,
            margin=dict(l=40, r=40, t=60, b=40),
            coloraxis_colorbar=dict(title="Коэффициент")
        )
        st.plotly_chart(fig, use_container_width=True)
