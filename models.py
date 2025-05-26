import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import RF_PARAM_GRID, MODEL_PATHS, RANDOM_STATE
from pipeline import DataPipeline
from visualization import Visualizer
from utils import io_tools
from utils.decorators import timeit, handle_errors
import plotly.express as px
import pandas as pd

"""
ModelManager — обучение, сравнение и сохранение моделей.

Поддерживает:
- Random Forest
- Linear Regression
"""


class ModelManager:
    def __init__(self):
        self.pipeline = DataPipeline()
        self.visualizer = Visualizer()

    @timeit
    @handle_errors
    def train_and_evaluate(self, df, target_col, model_name, log_transform, algorithm):
        """
               Обучает и оценивает выбранную модель, возвращает метрики и предсказания.

               Returns:
                   dict: Содержит модель, RMSE, MAE, y_test, y_pred, algorithm, key
               """
        df_clean = self.pipeline.preprocess(df, target_col)
        if df_clean.empty:
            st.error("Обработанный датасет пуст.")
            return None

        result = self.pipeline.split_data(df_clean, target_col, log_transform=log_transform)
        if result is None:
            st.error("❌ Не удалось разбить данные. Проверьте, что все признаки числовые.")
            return None

        X_train, X_test, y_train, y_test, scaler, features = result
        model = self.train_model(X_train, y_train, algorithm)
        rmse, mae, y_test, y_pred = self.evaluate_model(
            model, X_test, y_test, log_transform
        )

        model_key = f"{model_name}_{algorithm.lower().replace(' ', '_')}"
        st.session_state[f"model_{model_key}"] = model
        st.session_state[f"feature_names_{model_key}"] = features

        return {
            "model": model,
            "rmse": rmse,
            "mae": mae,
            "algorithm": algorithm,
            "key": model_key,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    @timeit
    def train_model(self, X_train, y_train, algorithm: str):
        if algorithm == "Random Forest":
            model = RandomForestRegressor(random_state=RANDOM_STATE)
            grid = GridSearchCV(
                model,
                RF_PARAM_GRID,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            st.success("🏁 Random Forest обучена")
            st.write("Лучшие параметры:", grid.best_params_)
            return grid.best_estimator_
        elif algorithm == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            st.success("🏁 Linear Regression обучена")
            return model
        else:
            st.error(f"Неизвестный алгоритм: {algorithm}")
            return None

    @timeit
    @handle_errors
    def evaluate_model(self, model, X_test, y_test, log_transform):
        y_pred = model.predict(X_test)
        if log_transform:
            y_pred = np.expm1(y_pred)
            y_test = np.expm1(y_test)

        y_pred = np.clip(y_pred, 0, 1e10)
        y_test = np.clip(y_test, 0, 1e10)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        st.info(f"📉 RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        return rmse, mae, y_test, y_pred

    def train_regression_ui(self, df, target_col, model_key, log_transform, title):
        st.subheader(title)
        if target_col not in df.columns:
            st.error("Целевая переменная отсутствует в данных.")
            return

        selected_algorithms = st.multiselect(
            "Выберите модели для сравнения",
            ["Random Forest", "Linear Regression"],
            default=["Random Forest"]
        )

        if st.button("🚀 Обучить и сравнить", key=f"train_{model_key}"):
            scores = []
            for algorithm in selected_algorithms:
                result = self.train_and_evaluate(df, target_col, model_key, log_transform, algorithm)
                if result:
                    scores.append(result)

            if scores:
                st.markdown("### 📊 Сравнение моделей")
                for r in scores:
                    st.markdown(f"**{r['algorithm']}**: RMSE = `{r['rmse']:.2f}`, MAE = `{r['mae']:.2f}`")
                    self.visualizer.plot_prediction_scatter(r["y_test"], r["y_pred"], r["key"])
                    self.visualizer.plot_feature_importance(r["model"], st.session_state[f"feature_names_{r['key']}"])

                # 📊 Итоговый bar plot по метрикам
                df_metrics = pd.DataFrame(scores)
                fig_rmse = px.bar(
                    df_metrics,
                    x="algorithm",
                    y="rmse",
                    title="RMSE по моделям",
                    labels={"rmse": "RMSE", "algorithm": "Модель"},
                    text_auto=".2f",
                    height=400,
                    color="algorithm"
                )
                st.plotly_chart(fig_rmse, use_container_width=True)

                fig_mae = px.bar(
                    df_metrics,
                    x="algorithm",
                    y="mae",
                    title="MAE по моделям",
                    labels={"mae": "MAE", "algorithm": "Модель"},
                    text_auto=".2f",
                    height=400,
                    color="algorithm"
                )
                st.plotly_chart(fig_mae, use_container_width=True)

    def save_model(self, key: str):
        for algo in ["Random Forest", "Linear Regression"]:
            model_key = f"{key}_{algo.lower().replace(' ', '_')}"
            model = st.session_state.get(f"model_{model_key}")
            if model:
                io_tools.save_model(model, MODEL_PATHS.get(model_key, f"models/model_{model_key}.pkl"))

    def load_model(self, key: str):
        for algo in ["Random Forest", "Linear Regression"]:
            model_key = f"{key}_{algo.lower().replace(' ', '_')}"
            model = io_tools.load_model(MODEL_PATHS.get(model_key, f"models/model_{model_key}.pkl"))
            if model:
                st.session_state[f"model_{model_key}"] = model

    def save_all(self):
        for key in ["ege", "niokr"]:
            self.save_model(key)

    def load_all(self):
        for key in ["ege", "niokr"]:
            self.load_model(key)
