import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import PLOTS_DIR
from utils import io_tools


class Visualizer:
    def __init__(self):
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

    def plot_prediction_scatter(self, y_true, y_pred, title):
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_xlabel("Фактические значения")
        ax.set_ylabel("Предсказанные значения")
        ax.set_title(f"{title} — предсказание vs факт")
        st.pyplot(fig)

        safe_title = title.replace(' ', '_')[:50]
        filename = f"{PLOTS_DIR}/regression_scatter_{safe_title}_{len(io_tools.saved_plots)}.png"
        self._save_plot(fig, filename, f"Сравнение факта и прогноза: {title}")

    def plot_feature_importance(self, model, features):
        if not hasattr(model, 'feature_importances'):
            st.warning("Модель не поддерживает важность признаков")
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x=model.feature_importances_, y=features, ax=ax, palette='viridis')
        ax.set_title("Важность признаков")
        st.pyplot(fig)

        filename = f"{PLOTS_DIR}/feature_importance_{len(io_tools.saved_plots)}.png"
        self._save_plot(fig, filename, "Важность признаков модели")

    def plot_correlation_heatmap(self, df):
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df.corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Тепловая карта корреляций")
        st.pyplot(fig)

        filename = f"{PLOTS_DIR}/correlation_heatmap_{len(io_tools.saved_plots)}.png"
        self._save_plot(fig, filename, "Тепловая карта корреляций")

    def _save_plot(self, fig, filename, description):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches='tight')
        io_tools.saved_plots.append(filename)
        io_tools.plot_descriptions.append(description)
        st.write(f"График сохранён как {filename}")
        plt.close(fig)
