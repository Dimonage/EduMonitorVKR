# utils/io_tools.py — функции для сохранения и загрузки данных и моделей

import streamlit as st
import pandas as pd
import joblib


@st.cache_data(show_spinner=False)
def load_dataframe(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        st.success(f"📄 Данные загружены из {path}")
        return df
    except Exception as e:
        st.error(f"❌ Ошибка загрузки данных: {e}")
        return None


def save_dataframe(df: pd.DataFrame, filename: str) -> bool:
    try:
        df.to_csv(filename, index=False)
        st.success(f"💾 Датафрейм сохранён: {filename}")
        return True
    except Exception as e:
        st.error(f"❌ Ошибка при сохранении: {e}")
        return False


def save_model(model, path: str):
    try:
        joblib.dump(model, path)
        st.success(f"🧠 Модель сохранена: {path}")
    except Exception as e:
        st.error(f"❌ Ошибка при сохранении модели: {e}")


def load_model(path: str):
    try:
        model = joblib.load(path)
        st.success(f"📦 Модель загружена из {path}")
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None
