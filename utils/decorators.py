# utils/decorators.py — универсальные декораторы

import streamlit as st
import time
from functools import wraps


def handle_errors(func):
    """Оборачивает функцию для обработки исключений с выводом в Streamlit."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Ошибка в '{func.__name__}': {e}")
            return None
    return wrapper


def timeit(func):
    """Измеряет и выводит время выполнения функции."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        st.info(f"⏱ Время выполнения '{func.__name__}': {duration:.2f} сек")
        return result
    return wrapper


def log_step(message: str):
    """Выводит сообщение перед выполнением функции."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            st.write(f"🔹 {message}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
