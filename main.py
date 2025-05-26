import streamlit as st

from state_manager import SessionStateManager
from config import EGE_TARGET, NIOKR_TARGET
from pipeline import DataPipeline
from models import ModelManager
from visualization import Visualizer
from utils import io_tools


def main() -> None:
    st.set_page_config(page_title="EduMonitor", layout="wide")
    st.title("🎓 Система анализа мониторинга вузов — EduMonitor")
    manager = SessionStateManager()
    # Инициализация состояния
    manager.initialize()

    # Обработка rerun-флага (гарантирует отображение интерфейса после загрузки)
    if st.session_state.get("trigger_rerun"):
        st.session_state.trigger_rerun = False
        st.rerun()

    # Инициализация компонентов
    data_pipeline = DataPipeline()
    model_manager = ModelManager()
    visualizer = Visualizer()

    # Загрузка и предобработка данных
    with st.expander("📂 Загрузка и предобработка данных", expanded=not st.session_state.data_loaded):
        df = data_pipeline.handle_data_upload()
        if df is not None:
            st.success("✅ Данные успешно загружены и обработаны.")
            st.write(df.head())
            st.session_state.trigger_rerun = True
            st.rerun()
        else:
            st.info("Загрузите файл формата .xlsx")

    if not st.session_state.data_loaded:
        st.warning("Пожалуйста, загрузите данные для продолжения.")
        return

    # Выбор задачи
    task = st.sidebar.selectbox(
        "Выберите задачу",
        [
            "Анализ данных",
            "Предсказание ЕГЭ",
            "Предсказание НИОКР",
            "Кластеризация",
            "Тепловая карта"
        ]
    )

    # Обработка задач
    if task == "Анализ данных":
        data_pipeline.render_data_analysis()

    elif task == "Предсказание ЕГЭ":
        model_manager.train_regression_ui(
            df=st.session_state.df_clean,
            target_col=EGE_TARGET,
            model_key="ege",
            log_transform=False,
            title="🔢 Предсказание среднего балла ЕГЭ"
        )

    elif task == "Предсказание НИОКР":
        model_manager.train_regression_ui(
            df=st.session_state.df_clean,
            target_col=NIOKR_TARGET,
            model_key="niokr",
            log_transform=True,
            title="🌐 Предсказание объема НИОКР"
        )

    elif task == "Кластеризация":
        data_pipeline.render_clustering_ui()

    elif task == "Тепловая карта":
        visualizer.plot_correlation_heatmap(st.session_state.df_clean)

    # Управление моделями
    with st.sidebar.expander("📊 Управление моделями"):
        if st.button("🔖 Сохранить все модели"):
            model_manager.save_all()
        if st.button("📂 Загрузить модели"):
            model_manager.load_all()

    # Экспорт отчета
    with st.sidebar.expander("📄 Экспорт отчета"):
        standard_text = st.text_area("Введите текст для отчета", height=150)
        if st.button("📅 Экспортировать в Word"):
            if io_tools.saved_plots:
                with st.spinner("Создание отчета..."):
                    success = io_tools.export_to_word(
                        output_file="edu_monitor_report.docx",
                        standard_text=standard_text or None
                    )
                    if success:
                        with open("edu_monitor_report.docx", "rb") as f:
                            st.download_button(
                                label="🔗 Скачать отчет",
                                data=f,
                                file_name="edu_monitor_report.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
            else:
                st.warning("Нет графиков для экспорта. Проведите анализ или визуализацию.")

    # Очистка визуализаций
    with st.sidebar.expander("❌ Очистка графиков"):
        if st.button("🧼 Очистить сохраненные графики"):
            io_tools.clear_saved_plots()


if __name__ == "__main__":
    main()
