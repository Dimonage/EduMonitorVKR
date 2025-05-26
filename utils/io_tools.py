# utils/io_tools.py — сохранение и загрузка моделей и таблиц

import streamlit as st
import pandas as pd
import joblib
import os

"""
Загрузка и сохранение моделей, датафреймов и кэшей.
"""


@st.cache_data
def save_dataframe(df: pd.DataFrame, filename: str) -> bool:
    """Сохраняет DataFrame в CSV."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        st.success(f"✅ Файл сохранён: {filename}")
        return True
    except Exception as e:
        st.error(f"❌ Ошибка при сохранении: {e}")
        return False


def save_model(model, path: str):
    """Сохраняет модель в файл .pkl."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        st.success(f"✅ Модель сохранена: {path}")
    except Exception as e:
        st.error(f"❌ Ошибка при сохранении модели: {e}")


def load_model(path: str):
    """Загружает модель из файла .pkl."""
    try:
        if not os.path.exists(path):
            st.warning(f"⚠️ Модель {path} не найдена.")
            return None
        model = joblib.load(path)
        st.success(f"✅ Модель загружена: {path}")
        return model
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {e}")
        return None
