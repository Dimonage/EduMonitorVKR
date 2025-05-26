import streamlit as st

from config import EGE_TARGET, NIOKR_TARGET
from state_manager import SessionStateManager
from pipeline import DataPipeline
from models import ModelManager
from visualization import Visualizer
from utils import globals as g
from utils.report_generator import export_to_word
from utils.ui_components import render_data_exploration_ui, render_clustering_visuals


def main():
    st.set_page_config(page_title="EduMonitor", layout="wide")
    st.title("🎓 EduMonitor: Анализ мониторинга вузов")

    # Инициализация сессии
    SessionStateManager.initialize()

    # Инициализация компонентов
    pipeline = DataPipeline()
    models = ModelManager()
    visualizer = Visualizer()

    # Блок загрузки данных
    with st.expander("📂 Загрузка и предобработка", expanded=not st.session_state.data_loaded):
        df = pipeline.handle_data_upload()
        if df is not None:
            st.success("✅ Данные загружены")
            st.write(df.head())

    if not st.session_state.data_loaded:
        st.info("🔄 Пожалуйста, загрузите файл .xlsx")
        return

    # Выбор действия
    task = st.sidebar.radio("Навигация", [
        "Анализ данных",
        "Предсказание ЕГЭ",
        "Предсказание НИОКР",
        "Кластеризация",
        "Тепловая карта"
    ])

    # Выполнение выбранной задачи
    df_clean = st.session_state.get("df_clean")

    if task == "Анализ данных":
        render_data_exploration_ui(df_clean)

    elif task == "Предсказание ЕГЭ":
        models.train_regression_ui(df_clean, EGE_TARGET, "ege", log_transform=False,
                                   title="📊 Прогноз среднего балла ЕГЭ")

    elif task == "Предсказание НИОКР":
        models.train_regression_ui(df_clean, NIOKR_TARGET, "niokr", log_transform=True, title="🧪 Прогноз объема НИОКР")



    elif task == "Кластеризация":

        st.subheader("🔗 Кластеризация вузов")

        n_clusters = st.slider("Количество кластеров", 2, 10, 3)

        if st.button("📊 Кластеризовать"):
            clustered_df, clusters = pipeline.clusterize(df_clean, n_clusters)

            st.session_state["cluster_df"] = clustered_df

            st.session_state["clusters"] = clusters

            st.session_state["cluster_ready"] = True

        # если кластеризация уже была выполнена

        if st.session_state.get("cluster_ready"):
            render_clustering_visuals(

                st.session_state["cluster_df"],

                st.session_state["clusters"]

            )

    elif task == "Тепловая карта":
        visualizer.plot_correlation_heatmap(df_clean)

    # Управление моделями
    with st.sidebar.expander("⚙️ Управление моделями"):
        if st.button("💾 Сохранить все модели"):
            models.save_all()
        if st.button("📂 Загрузить модели"):
            models.load_all()

    # Экспорт отчета
    with st.sidebar.expander("📄 Экспорт отчета"):
        text = st.text_area("Текст отчета", height=150)
        if st.button("📥 Сформировать Word"):
            if g.saved_plots:
                if export_to_word("edu_monitor_report.docx", text or None):
                    with open("edu_monitor_report.docx", "rb") as f:
                        st.download_button("📎 Скачать отчет", f, file_name="edu_monitor_report.docx")
            else:
                st.warning("Нет графиков для экспорта")

    # Очистка графиков
    with st.sidebar.expander("🧹 Очистка"):
        if st.button("Очистить графики"):
            g.saved_plots.clear()
            g.plot_descriptions.clear()
            st.success("Графики очищены")


if __name__ == "__main__":
    main()
