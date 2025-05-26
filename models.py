import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import RF_PARAM_GRID, MODEL_PATHS, RANDOM_STATE
from pipeline import DataPipeline
from visualization import Visualizer
from utils import io_tools


class ModelManager:
    def __init__(self):
        self.pipeline = DataPipeline()
        self.visualizer = Visualizer()

    def train_and_evaluate(self, df, target_col, model_name, log_transform):
        try:
            df_clean = self.pipeline.preprocess(df, target_col)
            if df_clean.empty:
                st.error("Обработанный датасет пуст.")
                return None

            X_train, X_test, y_train, y_test, scaler, features = self.pipeline.split_data(
                df_clean, target_col, log_transform=log_transform
            )

            model = self.train_model(X_train, y_train)
            rmse, mae = self.evaluate_model(
                model, X_test, y_test, target_col, features, log_transform
            )

            st.session_state[f"model_{model_name}"] = model
            st.session_state[f"feature_names_{model_name}"] = features
            return {
                "model": model,
                "rmse": rmse,
                "mae": mae
            }
        except Exception as e:
            st.error(f"Ошибка при обучении модели: {e}")
            return None

    def train_model(self, X_train, y_train):
        model = RandomForestRegressor(random_state=RANDOM_STATE)
        grid = GridSearchCV(
            model,
            RF_PARAM_GRID,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        st.write("🏁 Лучшие параметры:", grid.best_params_)
        return grid.best_estimator_

    def evaluate_model(self, model, X_test, y_test, name, features, log_transform):
        y_pred = model.predict(X_test)
        if log_transform:
            y_pred = np.expm1(y_pred)
            y_test = np.expm1(y_test)

        y_pred = np.clip(y_pred, 0, 1e10)
        y_test = np.clip(y_test, 0, 1e10)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        st.write(f"📉 RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        self.visualizer.plot_prediction_scatter(y_test, y_pred, name)
        self.visualizer.plot_feature_importance(model, features)

        return rmse, mae

    def train_regression_ui(self, df, target_col, model_key, log_transform, title):
        st.subheader(title)

        if target_col not in df.columns:
            st.error("Целевая переменная отсутствует в данных.")
            return

        trigger_key = f"train_{model_key}_requested"

        if st.button("🚀 Обучить модель", key=f"train_btn_{model_key}"):
            st.session_state[trigger_key] = True

        if st.session_state.get(trigger_key):
            result = self.train_and_evaluate(df, target_col, model_key, log_transform)
            if result:
                self.save_model(model_key)
            st.session_state[trigger_key] = False  # сбрасываем, чтобы не срабатывало бесконечно

    def save_model(self, key: str):
        model = st.session_state.get(f"model_{key}")
        if model:
            io_tools.save_model(model, MODEL_PATHS[key])

    def load_model(self, key: str):
        model = io_tools.load_model(MODEL_PATHS[key])
        if model:
            st.session_state[f"model_{key}"] = model
            return model
        return None

    def save_all(self):
        for key in MODEL_PATHS:
            self.save_model(key)

    def load_all(self):
        for key in MODEL_PATHS:
            self.load_model(key)
