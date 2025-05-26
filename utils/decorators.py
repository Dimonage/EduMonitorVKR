# utils/decorators.py — вспомогательные декораторы

import streamlit as st
import time
from functools import wraps

"""
Декораторы: замер времени и обработка ошибок с выводом в Streamlit.
"""


def timeit(func):
    """⏱️ Декоратор для измерения времени выполнения функции"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        st.info(f"⏱ Выполнено за {duration:.2f} сек")
        return result

    return wrapper


def handle_errors(func):
    """❌ Декоратор для обработки и отображения ошибок в Streamlit"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Ошибка в '{func.__name__}': {e}")
            return None

    return wrapper
