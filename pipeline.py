import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold

from config import RANDOM_STATE, DATA_PATHS, DEFAULT_TEST_SIZE
from utils.io_tools import save_dataframe
from utils.decorators import timeit, handle_errors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


@st.cache_data
def load_excel_file(uploaded_file):
    return pd.read_excel(uploaded_file)


"""
Модуль DataPipeline — для предобработки, фильтрации и кластеризации данных.
"""


class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.selector = VarianceThreshold(threshold=0.01)

    @handle_errors
    @timeit
    def handle_data_upload(self) -> pd.DataFrame | None:
        """
        Загружает Excel-файл через интерфейс Streamlit и выполняет его предобработку.

        Файл очищается от пустых строк и дубликатов, сохраняется в `session_state`.

        Returns:
            pd.DataFrame | None: Очищенный датафрейм или None, если загрузка не удалась.
        """
        uploaded_file = st.file_uploader("Загрузите файл (.xlsx)", type=["xlsx"], key="file_uploader")
        if uploaded_file:
            df = load_excel_file(uploaded_file)
            st.success(f"✅ Загружен: {uploaded_file.name} ({df.shape[0]} строк, {df.shape[1]} столбцов)")
            df_clean = self.preprocess(df)
            if not df_clean.empty:
                st.session_state.df_clean = df_clean
                st.session_state.data_loaded = True
                save_dataframe(df_clean, DATA_PATHS["processed"])
                st.rerun()
                return df_clean
            else:
                st.warning("⚠️ После очистки DataFrame оказался пустым")
        return None

    @handle_errors
    @timeit
    def preprocess(self, df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
        """
        Выполняет базовую очистку датафрейма:
        - удаление пустых строк и столбцов,
        - удаление дубликатов,
        - опционально — удаление строк с NaN в целевой переменной.

        Args:
            df (pd.DataFrame): Исходный датафрейм.
            target_col (str | None): Название целевого признака (для удаления NaN).

        Returns:
            pd.DataFrame: Очищенные данные.
        """
        df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
        df = df.drop_duplicates()
        if target_col and target_col in df.columns:
            df = df.dropna(subset=[target_col])
        return df

    @handle_errors
    @timeit
    def split_data(self, df: pd.DataFrame, target_col: str, log_transform: bool = False):
        if target_col not in df.columns:
            st.error("Целевая переменная не найдена.")
            return None

        X = df.drop(columns=[target_col])
        y = df[target_col]

        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        if log_transform:
            y = y.apply(lambda v: np.log1p(v) if v > 0 else np.nan).dropna()
            X = X.loc[y.index]  # подгонка X под y

        # Преобразователь признаков
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ])

        X_encoded = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE
        )

        # Сохраняем имена признаков (опционально)
        feature_names = (
                num_cols +
                preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
        )

        return X_train, X_test, y_train, y_test, preprocessor, feature_names

    @handle_errors
    @timeit
    def clusterize(self, df: pd.DataFrame, n_clusters: int = 3):
        """
        Выполняет кластеризацию методом KMeans по числовым признакам.

        Args:
            df (pd.DataFrame): Данные для кластеризации.
            n_clusters (int): Количество кластеров.

        Returns:
            tuple: Кластеризованный DataFrame и массив меток кластеров.
        """
        X = df.select_dtypes(include=['number'])
        X_scaled = self.scaler.fit_transform(X)
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        clusters = model.fit_predict(X_scaled)

        df_clustered = df.copy()
        df_clustered['Кластер'] = clusters
        st.session_state.df_clustered = df_clustered
        st.session_state.clusters = clusters
        st.success("✅ Кластеризация выполнена")
        save_dataframe(df_clustered, DATA_PATHS["filtered"])
        return df_clustered, clusters
