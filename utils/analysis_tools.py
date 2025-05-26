# utils/analysis_tools.py — очистка сохранённых графиков и их описаний

import streamlit as st
import os
from config import PLOTS_DIR


def clear_saved_plots():
    """Удаляет все сохранённые графики и очищает session_state"""
    try:
        # Удаляем все файлы из папки plots/
        if os.path.exists(PLOTS_DIR):
            for file in os.listdir(PLOTS_DIR):
                file_path = os.path.join(PLOTS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Очищаем списки в session_state
        st.session_state.saved_plots = []
        st.session_state.plot_descriptions = []

        st.success("🧼 Все графики и описания удалены.")
        return True

    except Exception as e:
        st.error(f"❌ Ошибка при очистке графиков: {e}")
        return False
