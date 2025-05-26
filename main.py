import streamlit as st

from state_manager import SessionStateManager
from config import EGE_TARGET, NIOKR_TARGET
from pipeline import DataPipeline
from models import ModelManager
from visualization import Visualizer

from utils.ui_components import render_data_exploration_ui, render_clustering_visuals
from utils.report_generator import export_to_word
from utils.analysis_tools import clear_saved_plots


def main():
    st.set_page_config(page_title="EduMonitor", layout="wide")
    st.title("🎓 EduMonitor — Анализ мониторинга вузов")

    # Инициализация состояния
    SessionStateManager.initialize()
    pipeline = DataPipeline()
    models = ModelManager()
    visualizer = Visualizer()

    # --- Загрузка данных ---
    if not st.session_state.data_loaded:
        with st.expander("📂 Загрузка и предобработка", expanded=True):
            df = pipeline.handle_data_upload()
            if df is not None:
                st.success("✅ Данные успешно загружены.")
                # После загрузки ставим флаг, чтобы не загружать повторно
                st.session_state.data_loaded = True
                st.session_state.df_clean = df
                st.rerun()
        return  # Прекращаем выполнение до загрузки данных

    # --- После загрузки данные доступны ---
    # Копируем для фильтрации
    df_filtered = st.session_state.df_clean.copy()

    # --- Фильтрация по категориальным столбцам ---
    st.sidebar.subheader("🔍 Фильтрация данных")
    cat_cols = df_filtered.select_dtypes(include=["object", "category", "bool"]).columns

    for col in cat_cols:
        unique_vals = df_filtered[col].dropna().unique().tolist()
        if 1 < len(unique_vals) < 50:
            selected_vals = st.sidebar.multiselect(f"{col}", unique_vals, default=unique_vals)
            if selected_vals:
                df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]

    st.session_state.df_filtered = df_filtered

    # --- Выбор задачи ---
    task = st.sidebar.selectbox("Выберите задачу", [
        "Анализ данных",
        "Предсказание ЕГЭ",
        "Предсказание НИОКР",
        "Кластеризация",
        "Тепловая карта"
    ])

    # --- Основной функционал ---
    if task == "Анализ данных":
        render_data_exploration_ui(df_filtered)

    elif task == "Предсказание ЕГЭ":
        models.train_regression_ui(
            df=df_filtered,
            target_col=EGE_TARGET,
            model_key="ege",
            log_transform=False,
            title="🔢 Предсказание среднего балла ЕГЭ"
        )

    elif task == "Предсказание НИОКР":
        models.train_regression_ui(
            df=df_filtered,
            target_col=NIOKR_TARGET,
            model_key="niokr",
            log_transform=True,
            title="🌐 Предсказание объема НИОКР"
        )

    elif task == "Кластеризация":
        with st.expander("📊 Кластеризация вузов", expanded=True):
            n_clusters = st.slider("Количество кластеров", 2, 10, 3)
            if st.button("📍 Кластеризовать"):
                df_clustered, clusters = pipeline.clusterize(df_filtered, n_clusters)
                render_clustering_visuals(df_clustered, clusters)

    elif task == "Тепловая карта":
        visualizer.plot_correlation_heatmap(df_filtered)

    # --- Управление моделями ---
    with st.sidebar.expander("📦 Модели"):
        if st.button("💾 Сохранить модели"):
            models.save_all()
        if st.button("📂 Загрузить модели"):
            models.load_all()

    # --- Экспорт отчета ---
    with st.sidebar.expander("📄 Экспорт в Word"):
        user_text = st.text_area("Комментарий к отчету", height=150)
        if st.button("📝 Сформировать отчет"):
            plots = st.session_state.get("saved_plots", [])
            if plots:
                if export_to_word("edu_monitor_report.docx", user_text or None):
                    with open("edu_monitor_report.docx", "rb") as f:
                        st.download_button(
                            label="📥 Скачать отчет",
                            data=f,
                            file_name="edu_monitor_report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
            else:
                st.warning("⚠️ Нет графиков для экспорта. Сначала постройте визуализации.")

    # --- Очистка графиков ---
    with st.sidebar.expander("🧼 Очистка графиков"):
        if st.button("🗑 Очистить сохранённые графики"):
            clear_saved_plots()


if __name__ == "__main__":
    main()
