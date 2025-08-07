from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class ThemeTokens:
    color_bg: str = "#0e1117"
    color_surface: str = "#111827"
    color_border: str = "#242b36"
    color_text: str = "#e6edf3"
    color_text_muted: str = "#9ba3af"
    color_accent: str = "#3b82f6"
    color_success: str = "#22c55e"
    color_warning: str = "#f59e0b"
    color_error: str = "#ef4444"
    radius_sm: str = "6px"
    radius_md: str = "10px"
    radius_lg: str = "12px"
    spacing_sm: str = "6px"
    spacing_md: str = "10px"
    spacing_lg: str = "16px"


class ThemeManager:
    """Loads and applies a QSS theme with simple token interpolation."""

    tokens: ThemeTokens

    def __init__(self, tokens: ThemeTokens | None = None) -> None:
        self.tokens = tokens or ThemeTokens()

    def _interpolate(self, qss: str) -> str:
        mapping: Dict[str, str] = self.tokens.__dict__  # type: ignore
        for key, value in mapping.items():
            qss = qss.replace(f"{{{{{key}}}}}", str(value))
        return qss

    def apply(self, app_or_widget: QtWidgets.QApplication | QtWidgets.QWidget) -> None:
        file_path = QtCore.QFile("assets/theme.qss")
        if file_path.exists() and file_path.open(QtCore.QIODevice.ReadOnly | QtCore.QIODevice.Text):
            try:
                raw = bytes(file_path.readAll()).decode("utf-8")
            finally:
                file_path.close()
            qss = self._interpolate(raw)
            if isinstance(app_or_widget, QtWidgets.QApplication):
                app_or_widget.setStyleSheet(qss)
            else:
                app_or_widget.setStyleSheet(qss)
        else:
            # Fallback: apply nothing if theme file missing
            pass


def setup_pre_qapp() -> None:
    """Call BEFORE creating QApplication: HiDPI rounding policy and pixmaps attribute."""
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)


def setup_hidpi_and_font(app: QtWidgets.QApplication) -> None:
    """Set a modern default font if available (AFTER QApplication exists)."""

    # Try to load Inter locally; otherwise fall back to system
    font_loaded = False
    try:
        font_id = QtGui.QFontDatabase.addApplicationFont("assets/fonts/Inter-Variable.ttf")
        if font_id != -1:
            families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
            if families:
                app.setFont(QtGui.QFont(families[0], 12))
                font_loaded = True
    except Exception:
        pass

    if not font_loaded:
        # macOS often has SF Pro; otherwise use Helvetica Neue
        for family in ["SF Pro Text", "Inter", "Helvetica Neue", "Arial"]:
            f = QtGui.QFont(family, 12)
            if QtGui.QFontInfo(f).family():
                app.setFont(f)
                break


