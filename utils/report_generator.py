from docx import Document
from docx.shared import Inches
from globals import saved_plots, plot_descriptions
import os
import streamlit as st


def export_to_word(filename, text):
    doc = Document()
    doc.add_heading("Отчёт EduMonitor", 0)
    doc.add_paragraph(text or "Отчёт содержит визуализации и результаты анализа.")
    doc.add_heading("Визуализации", level=1)

    for path, desc in zip(saved_plots, plot_descriptions):
        if os.path.exists(path):
            doc.add_paragraph(desc)
            doc.add_picture(path, width=Inches(6))
    doc.save(filename)
    st.success(f"📄 Отчёт сохранён: {filename}")
