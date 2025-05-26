# visualization.py — визуализация метрик и корреляций

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from config import PLOTS_DIR
from utils.decorators import timeit, handle_errors


class Visualizer:

    def __init__(self):
        os.makedirs(PLOTS_DIR, exist_ok=True)

    @handle_errors
    @timeit
    def plot_prediction_scatter(self, y_true, y_pred, label: str):
        if y_true is None or y_pred is None:
            st.warning("Данные для scatter plot отсутствуют.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_true, y_pred, alpha=0.6, color='mediumseagreen')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_xlabel("Фактические значения", fontsize=12)
        ax.set_ylabel("Предсказанные значения", fontsize=12)
        ax.set_title(f"Сравнение предсказания и факта: {label}", fontsize=14)
        ax.tick_params(labelsize=10)
        self._render_and_save(fig, f"scatter_{label}.png", f"Предсказание vs Факт: {label}")

    @handle_errors
    @timeit
    def plot_feature_importance(self, model, feature_names: list[str]):
        if not hasattr(model, "feature_importances_"):
            st.info("Модель не поддерживает важность признаков.")
            return

        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        features = [feature_names[i] for i in sorted_idx[:10]]
        scores = importances[sorted_idx[:10]]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=scores, y=features, ax=ax, color="skyblue")
        ax.set_title("Топ-10 важных признаков", fontsize=14)
        ax.set_xlabel("Важность", fontsize=12)
        ax.set_ylabel("Признаки", fontsize=12)
        ax.tick_params(labelsize=10)
        self._render_and_save(fig, "feature_importance.png", "Топ-10 признаков модели")

    @handle_errors
    @timeit
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        corr = df.select_dtypes(include=["number"]).corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Корреляционная матрица", fontsize=14)
        ax.tick_params(labelsize=10)
        self._render_and_save(fig, "correlation_matrix.png", "Корреляционная матрица")

    def _render_and_save(self, fig, filename: str, description: str):
        path = os.path.join(PLOTS_DIR, filename)
        fig.tight_layout(pad=2.0)
        st.pyplot(fig)
        fig.savefig(path)

        # ✅ Используем session_state
        if "saved_plots" not in st.session_state:
            st.session_state.saved_plots = []
        if "plot_descriptions" not in st.session_state:
            st.session_state.plot_descriptions = []

        st.session_state.saved_plots.append(path)
        st.session_state.plot_descriptions.append(description)
        plt.close(fig)
