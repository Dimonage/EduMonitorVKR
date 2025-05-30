# 🎓 EduMonitor — Интеллектуальная система анализа данных вузов

**EduMonitor** — это web-приложение на Python + Streamlit для анализа мониторинга вузов. Включает:

* предобработку данных,
* обучение моделей (ЕГЭ и НИОКР),
* кластеризацию вузов,
* визуализации и экспорт отчёта.

---

## 🚀 Быстрый старт

### 🔧 Установка

```bash
git clone https://github.com/Dimonage/edumonitor.git
cd edumonitor
pip install -r requirements.txt
```

### ▶️ Запуск

```bash
streamlit run main.py
```

---

## 🧩 Структура проекта

```
edumonitor/
├── main.py                         # 🎛 Главный интерфейс Streamlit (обновлённый)
├── config.py                       # ⚙️ Константы: целевые столбцы, пути, параметры моделей
├── pipeline.py                     # 🔄 Загрузка, предобработка, кластеризация, split_data
├── models.py                       # 🤖 Обучение моделей, сравнение RandomForest и LinearRegression
├── visualization.py                # 📊 Визуализация результатов (plotly и matplotlib)
├── state_manager.py                # 🧠 Инициализация session_state и переменных

├── requirements.txt                # 📦 Зависимости проекта

├── data/                           # 📂 Данные
│   ├── processed_data.csv          # Данные после загрузки и очистки
│   ├── filtered_data.csv           # Отфильтрованные по UI
│   └── filtered_data_export.csv    # Экспортируемая версия (по кнопке)

├── models/                         # 🧠 Сохранённые модели
│   ├── model_ege.pkl               # Random Forest для ЕГЭ
│   ├── model_niokr.pkl             # Random Forest для НИОКР
│   ├── model_ege_linear.pkl        # Linear Regression для ЕГЭ
│   └── model_niokr_linear.pkl      # Linear Regression для НИОКР

├── plots/                          # 🖼️ Сохранённые графики для экспорта
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   ├── scatter_ege.png
│   └── ...

├── edu_monitor_report.docx         # 📄 Сформированный отчёт Word

└── utils/                          # 🔧 Утилитарные модули
    ├── __init__.py                 # (пустой, но делает папку модулем)
    ├── io_tools.py                 # 💾 Сохранение/загрузка моделей и файлов
    ├── decorators.py               # ⏱️ Декораторы: @timeit, @handle_errors
    ├── report_generator.py         # 📝 Генерация .docx отчета (работает через session_state)
    ├── analysis_tools.py           # 🧹 Очистка графиков, вспомогательные аналитические функции
    └── ui_components.py            # 🧩 Визуальные компоненты UI (фильтрация, графики с plotly)
```

---

## ⚙️ Функциональность

### 1. 📂 Загрузка и предобработка данных

* Загрузка `.xlsx` файла
* Очистка пропусков, дубликатов, выбросов
* Масштабирование признаков
* Автоматическое логарифмирование (по необходимости).

### 2. 📈 Анализ и визуализация

* Проверка пропусков, дубликатов.
* Гистограммы распределения.
* Характеристики признаков.
* Корреляции между переменными.
* Фильтрация по категориям (в боковой панели)

### 3. 🤖 Предсказание:

#### 📊 ЕГЭ (`model_ege.pkl`)

* Целевая переменная: средний балл ЕГЭ студентов.
* Используется `RandomForestRegressor + GridSearchCV`.

#### 🧪 НИОКР (`model_niokr.pkl`)

* Целевая переменная: объём НИОКР.
* Модель с логарифмированием и оценкой по RMSE, MAE

### 4. 🧠 Кластеризация

* Метод: `KMeans`.
* Визуализации:

  * Диаграмма рассеяния (scatter)
  * Boxplot по кластерам
  * Гистограмма распределения

### 5. 🗺️ Тепловая карта

* Автоматическое построение корреляционной матрицы
* Возможность выбора топ-N признаков

### 6. 📄 Генерация отчета

* Формат: .docx 
* Включает все построенные графики 
* Пользователь может добавить текстовый комментарий 
* Отчёт скачивается через интерфейс

---

## 📦 Зависимости (requirements.txt)

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
python-docx
joblib
openpyxl
```
