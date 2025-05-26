# visualization.py — визуализация метрик и корреляций

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from utils.decorators import timeit, handle_errors
from utils.globals import saved_plots, plot_descriptions
from config import PLOTS_DIR


class Visualizer:

    def __init__(self):
        os.makedirs(PLOTS_DIR, exist_ok=True)

    @handle_errors
    @timeit
    def plot_prediction_scatter(self, y_true, y_pred, label: str):
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.6, color='mediumseagreen')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_xlabel("Фактические значения")
        ax.set_ylabel("Предсказанные значения")
        ax.set_title(f"Сравнение предсказания и факта: {label}")
        self._render_and_save(fig, f"scatter_{label}.png", f"Предсказание vs Факт: {label}")

    @handle_errors
    @timeit
    def plot_feature_importance(self, model, feature_names: list[str]):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        features = [feature_names[i] for i in sorted_idx[:10]]
        scores = importances[sorted_idx[:10]]

        fig, ax = plt.subplots()
        sns.barplot(x=scores, y=features, ax=ax, palette="viridis")
        ax.set_title("Топ-10 важных признаков")
        ax.set_xlabel("Важность")
        self._render_and_save(fig, "feature_importance.png", "Топ-10 признаков модели")

    @handle_errors
    @timeit
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        corr = df.select_dtypes(include=["number"]).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Корреляционная матрица")
        self._render_and_save(fig, "correlation_matrix.png", "Корреляционная матрица")

    def _render_and_save(self, fig, filename: str, description: str):
        path = os.path.join(PLOTS_DIR, filename)
        fig.tight_layout()
        st.pyplot(fig)
        fig.savefig(path)
        saved_plots.append(path)
        plot_descriptions.append(description)
        plt.close(fig)
