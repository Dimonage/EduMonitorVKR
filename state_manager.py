# state_manager.py — инициализация состояния Streamlit

import streamlit as st


class SessionStateManager:
    """Класс для инициализации и управления состоянием сессии"""

    @staticmethod
    def initialize():
        """Инициализация ключевых переменных в session_state"""
        defaults = {
            "data_loaded": False,
            "df_clean": None,
            "trigger_rerun": False,
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Инициализация флагов для моделей
        for model_key in ["ege", "niokr", "kmeans"]:
            if f"model_{model_key}" not in st.session_state:
                st.session_state[f"model_{model_key}"] = None
            if f"{model_key}_trained" not in st.session_state:
                st.session_state[f"{model_key}_trained"] = False
