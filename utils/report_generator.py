# utils/report_generator.py — экспорт отчёта в Word

import os
import streamlit as st
from docx import Document
from docx.shared import Inches
from config import REPORT_PATH
from utils.globals import saved_plots, plot_descriptions


def export_to_word(output_file: str = REPORT_PATH, standard_text: str | None = None) -> bool:
    """Создаёт отчёт .docx с графиками и текстом"""
    try:
        doc = Document()

        # Заголовок и введение
        doc.add_heading("📊 Отчёт по анализу данных мониторинга вузов", level=0)
        doc.add_paragraph(
            standard_text or
            "В этом отчёте представлены результаты визуализации, кластеризации и предсказаний, выполненных системой EduMonitor."
        )

        doc.add_heading("📈 Визуализации", level=1)

        # Вставка всех сохранённых графиков
        for path, desc in zip(saved_plots, plot_descriptions):
            if os.path.exists(path):
                doc.add_paragraph(desc)
                doc.add_picture(path, width=Inches(6))

        # Сохранение
        doc.save(output_file)
        st.success(f"📄 Отчёт успешно сохранён: {output_file}")
        return True

    except Exception as e:
        st.error(f"❌ Ошибка создания отчёта: {e}")
        return False
