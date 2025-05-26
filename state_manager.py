import streamlit as st


class SessionStateManager:
    """
    Менеджер состояния Streamlit.
    Инициализирует все ключи session_state при запуске приложения.
    """

    DEFAULT_KEYS = {
        "df_clean": None,
        "clusters": None,
        "kmeans": None,
        "model_ege": None,
        "model_niokr": None,
        "feature_names_ege": None,
        "feature_names_niokr": None,
        "data_loaded": False
    }

    def initialize(self) -> None:
        """
        Инициализирует session_state со значениями по умолчанию,
        если они ещё не заданы.
        """
        for key, value in SessionStateManager.DEFAULT_KEYS.items():
            if key not in st.session_state:
                st.session_state[key] = value
