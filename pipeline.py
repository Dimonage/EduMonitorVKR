import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans

from config import N_FEATURES_SELECT, DEFAULT_TEST_SIZE, RANDOM_STATE
from utils import ui_components, io_tools


class DataPipeline:
    def handle_data_upload(self) -> pd.DataFrame | None:
        uploaded_file = st.file_uploader("Загрузите датасет (.xlsx)", type=["xlsx"], key="file_uploader")

        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                st.success(f"📥 Датасет загружен. Размер: {df.shape}")
                df_clean = self.preprocess(df)
                if not df_clean.empty:
                    st.session_state.df_clean = df_clean
                    st.session_state.data_loaded = True
                    return df_clean
                else:
                    st.warning("⚠️ Обработка вернула пустой DataFrame.")
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке: {e}")
        return None

    def preprocess(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        df_clean = df_clean.loc[:, df_clean.var() > 0]

        if target_col and target_col in df_clean.columns:
            df_clean[target_col] = df_clean[target_col].clip(lower=1e-6)
            outliers = io_tools.detect_outliers(df_clean, target_col)
            if outliers is not None:
                df_clean = df_clean[~outliers]
            df_clean = df_clean.dropna(subset=[target_col])

        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
        st.write(f"Данные обработаны. Размер: {df_clean.shape}")
        return df_clean

    def split_data(self, df: pd.DataFrame, target_col: str, log_transform: bool = False):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if log_transform:
            y = np.log1p(y)
            st.write("Применено логарифмирование к целевой переменной")

        X_selected, selected_features = self.select_features(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1e6, neginf=-1e6)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, selected_features

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = N_FEATURES_SELECT):
        try:
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            st.write(f"Выбраны признаки: {selected_features}")
            return X[selected_features], selected_features
        except Exception as e:
            st.error(f"Ошибка при отборе признаков: {e}")
            return X, X.columns.tolist()

    def clusterize(self, df: pd.DataFrame, n_clusters: int = 3):
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)

            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
            clusters = kmeans.fit_predict(X_scaled)

            return clusters, kmeans, scaler
        except Exception as e:
            st.error(f"Ошибка кластеризации: {e}")
            return None, None, None

    def render_clustering_ui(self):
        st.subheader("📊 Кластеризация вузов")
        if st.button("Выполнить кластеризацию"):
            clusters, model, _ = self.clusterize(st.session_state.df_clean)
            if clusters is not None:
                st.session_state.clusters = clusters
                st.session_state.kmeans = model
                st.session_state.df_clean['Кластер'] = clusters
                io_tools.save_model(model, "models/model_kmeans.pkl")
                io_tools.save_dataframe(st.session_state.df_clean, "data/clustered_data.csv")

        if st.session_state.clusters is not None:
            ui_components.render_clustering_visuals(st.session_state.df_clean, st.session_state.clusters)

    def render_data_analysis(self):
        st.subheader("📈 Анализ данных")
        df = st.session_state.get("df_clean")

        if df is None or df.empty:
            st.error("Нет данных для анализа. Пожалуйста, загрузите и обработайте данные.")
            return

        ui_components.render_data_exploration_ui(df)
