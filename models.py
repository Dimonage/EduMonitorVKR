# models.py — обучение и оценка моделей

import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import RF_PARAM_GRID, MODEL_PATHS, RANDOM_STATE
from pipeline import DataPipeline
from visualization import Visualizer
from utils.io_tools import save_model, load_model
from utils.decorators import timeit, handle_errors


class ModelManager:
    def __init__(self):
        self.pipeline = DataPipeline()
        self.visualizer = Visualizer()

    @handle_errors
    @timeit
    def train_and_evaluate(
            self,
            df,
            target_col: str,
            model_key: str,
            log_transform: bool
    ) -> dict | None:
        """Обработка полного цикла обучения и оценки модели"""
        df_clean = self.pipeline.preprocess(df, target_col)
        if df_clean.empty:
            st.error("⚠️ Обработанный датасет пуст.")
            return None

        split_result = self.pipeline.split_data(df_clean, target_col, log_transform=log_transform)
        if split_result is None:
            st.error("❌ Не удалось разбить данные. Проверьте, что все признаки числовые.")
            return None

        X_train, X_test, y_train, y_test, scaler, features = split_result

        model = self.train_model(X_train, y_train)
        rmse, mae = self.evaluate_model(
            model, X_test, y_test, target_col, features, log_transform
        )

        st.session_state[f"model_{model_key}"] = model
        st.session_state[f"feature_names_{model_key}"] = features

        return {"model": model, "rmse": rmse, "mae": mae}

    @handle_errors
    @timeit
    def train_model(self, X_train, y_train):
        """Поиск лучшей модели через GridSearchCV"""
        model = RandomForestRegressor(random_state=RANDOM_STATE)
        grid = GridSearchCV(
            model,
            RF_PARAM_GRID,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        st.success("✅ Модель обучена")
        st.write("🔧 Лучшие параметры:", grid.best_params_)
        return grid.best_estimator_

    @handle_errors
    @timeit
    def evaluate_model(
        self,
        model,
        X_test,
        y_test,
        target_col: str,
        features: list[str],
        log_transform: bool
    ) -> tuple[float, float]:
        """Оценка качества модели и визуализация результатов"""
        y_pred = model.predict(X_test)

        if log_transform:
            y_pred = np.expm1(y_pred)
            y_test = np.expm1(y_test)

        y_pred = np.clip(y_pred, 0, 1e10)
        y_test = np.clip(y_test, 0, 1e10)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        st.success(f"📊 RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        self.visualizer.plot_prediction_scatter(y_test, y_pred, target_col)
        self.visualizer.plot_feature_importance(model, features)

        return rmse, mae

    def train_regression_ui(
        self,
        df,
        target_col: str,
        model_key: str,
        log_transform: bool,
        title: str
    ) -> None:
        """UI-обёртка над обучением модели"""
        st.subheader(title)

        if target_col not in df.columns:
            st.error("Целевая переменная отсутствует в данных.")
            return

        trigger_key = f"train_{model_key}_trigger"

        if st.button("🚀 Обучить модель", key=f"btn_{model_key}"):
            st.session_state[trigger_key] = True

        if st.session_state.get(trigger_key):
            result = self.train_and_evaluate(df, target_col, model_key, log_transform)
            if result:
                self.save_model(model_key)
                st.session_state[f"{model_key}_trained"] = True
            st.session_state[trigger_key] = False

    @handle_errors
    def save_model(self, model_key: str) -> None:
        model = st.session_state.get(f"model_{model_key}")
        if model:
            save_model(model, MODEL_PATHS[model_key])

    @handle_errors
    def load_model(self, model_key: str):
        model = load_model(MODEL_PATHS[model_key])
        if model:
            st.session_state[f"model_{model_key}"] = model
            return model
        return None

    def save_all(self) -> None:
        for key in MODEL_PATHS:
            self.save_model(key)

    def load_all(self) -> None:
        for key in MODEL_PATHS:
            self.load_model(key)
