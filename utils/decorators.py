import time
import streamlit as st
from functools import wraps


def timeit(func):
    """
    Декоратор для измерения времени выполнения функции
    и отображения его в интерфейсе Streamlit.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        st.info(f"⏱ {func.__name__} выполнена за {duration:.2f} сек")
        return result
    return wrapper


def handle_errors(func):
    """
    Декоратор для перехвата и отображения ошибок Streamlit'ом.
    Предотвращает крашинг интерфейса при сбое.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Ошибка в {func.__name__}: {e}")
            return None
    return wrapper
