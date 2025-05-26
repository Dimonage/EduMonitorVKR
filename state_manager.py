# state_manager.py — инициализация Streamlit session_state

import streamlit as st

"""
Модуль инициализации Streamlit session_state с ключами по умолчанию.
"""


class SessionStateManager:
    @staticmethod
    def initialize():
        """
              Устанавливает флаги session_state для загрузки, моделей и фильтрации.
              """
        defaults = {
            # 🔹 Состояние загрузки и фильтрации
            "data_loaded": False,
            "df_clean": None,
            "df_filtered": None,
            "df_clustered": None,
            "clusters": None,

            # 🔹 Графики для отчёта
            "saved_plots": [],
            "plot_descriptions": [],

            # 🔹 Флаги обученности моделей
            "model_ege_random_forest_trained": False,
            "model_ege_linear_regression_trained": False,
            "model_niokr_random_forest_trained": False,
            "model_niokr_linear_regression_trained": False,

            # 🔹 Флаги загрузки моделей с диска
            "model_ege_random_forest_loaded": False,
            "model_ege_linear_regression_loaded": False,
            "model_niokr_random_forest_loaded": False,
            "model_niokr_linear_regression_loaded": False
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
