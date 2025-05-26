# utils/__init__.py — автоматический импорт подмодулей утилит

from . import io_tools
from . import decorators
from . import report_generator
from . import analysis_tools
from . import ui_components

# ✅ Удобный доступ к часто используемым компонентам
from .io_tools import save_model, load_model, save_dataframe
from .decorators import timeit, handle_errors
from .analysis_tools import clear_saved_plots
from .report_generator import export_to_word
from .ui_components import (
    render_data_exploration_ui,
    render_clustering_visuals
)