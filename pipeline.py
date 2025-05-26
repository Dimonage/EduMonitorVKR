
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


@st.cache_data
def load_excel_file(uploaded_file):
    return pd.read_excel(uploaded_file)


class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.selector = VarianceThreshold(threshold=0.01)

    @handle_errors
    @timeit
    def handle_data_upload(self) -> pd.DataFrame | None:
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
        df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
        df = df.drop_duplicates()
        if target_col and target_col in df.columns:
            df = df.dropna(subset=[target_col])
        return df

    @handle_errors
    @timeit
    def split_data(self, df: pd.DataFrame, target_col: str, log_transform: bool = False):
        df = df.select_dtypes(include=[np.number])
        if target_col not in df.columns:
            st.error(f"Целевая переменная {target_col} отсутствует среди числовых столбцов.")
            return None

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.empty or y.empty:
            st.error("Недостаточно данных после фильтрации.")
            return None

        X_scaled = self.scaler.fit_transform(X)
        if log_transform:
            y = y.apply(lambda v: np.log1p(v) if v > 0 else np.nan).dropna()
            X_scaled = X_scaled[:len(y)]

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE
        )
        features = X.columns.tolist()
        return X_train, X_test, y_train, y_test, self.scaler, features

    @handle_errors
    @timeit
    def clusterize(self, df: pd.DataFrame, n_clusters: int = 3):
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
