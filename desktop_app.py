from __future__ import annotations

import sys
import html
from typing import Optional
import re
from datetime import datetime
import math

from PySide6 import QtCore, QtGui, QtWidgets
from ollama import chat as ollama_chat

import config
from system_prompts import SYSTEM_PROMPTS
from core.chat_service import ChatSession, REASONING_OPTIONS
from theme import ThemeManager, setup_hidpi_and_font, setup_pre_qapp

try:
    import qtawesome as qta  # type: ignore
except Exception:  # pragma: no cover - optional icons
    qta = None


def _load_pixmap(path: str, size: int = 28) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(path)
    if pm.isNull():
        # Placeholder
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        brush = QtGui.QBrush(QtGui.QColor(59, 130, 246))
        painter.setBrush(brush)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, size, size)
        painter.end()
    return pm


def _generate_user_avatar_pixmap(size: int = 28) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setBrush(QtGui.QBrush(QtGui.QColor(17, 24, 39)))
    painter.setPen(QtGui.QPen(QtGui.QColor(75, 85, 99), 1))
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    return pm


class AutoResizeTextBrowser(QtWidgets.QTextBrowser):
    """A text browser that grows vertically with its content (no inner scrollbars)."""

    def __init__(self) -> None:
        super().__init__()
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.document().contentsChanged.connect(self._update_height)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_height()

    def _update_height(self) -> None:
        doc = self.document()
        doc.setTextWidth(self.viewport().width())
        new_height = int(doc.size().height()) + 6
        self.setMinimumHeight(new_height)
        self.setMaximumHeight(new_height)


class GrowingPlainTextEdit(QtWidgets.QPlainTextEdit):
    """PlainTextEdit that grows with content and handles Cmd+Enter to send."""

    sendRequested = QtCore.Signal()

    def __init__(self, min_height: int = 44, max_height: int = 140) -> None:
        super().__init__()
        self._min_h = min_height
        self._max_h = max_height
        self.setFixedHeight(self._min_h)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.textChanged.connect(self.adjustHeight)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if (e.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter)) and (e.modifiers() & QtCore.Qt.KeyboardModifier.MetaModifier):
            e.accept()
            self.sendRequested.emit()
            return
        super().keyPressEvent(e)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.adjustHeight()

    def adjustHeight(self) -> None:
        fm = self.fontMetrics()
        line = fm.lineSpacing()
        # Estimate needed height using wrapped text bounding rect
        width = max(1, self.viewport().width() - 8)
        text = self.toPlainText()
        if text:
            rect = fm.boundingRect(0, 0, width, 1_000_000, QtCore.Qt.TextFlag.TextWordWrap, text)
            rows = max(1, math.ceil(rect.height() / max(1, line)))
        else:
            rows = 1
        rows = min(rows, 8)
        target = int(rows * line + 12)
        target = min(self._max_h, max(self._min_h, target))
        self.setFixedHeight(target)
        # Scrollbar only when overflow beyond max rows
        if rows >= 8 and text:
            self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        else:
            self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)


## Removed KaTeX WebEngine due to rendering issues. Falling back to QTextBrowser-based rendering.
class MessageRow(QtWidgets.QWidget):
    def __init__(self, role: str, title: str) -> None:
        super().__init__()
        self.role = role
        self._created_at = datetime.now()
        self._plain_accumulator: str = ""

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 8)
        root.setSpacing(8)

        # Left avatar column with fixed width so bubbles align regardless of avatar
        self.avatar_label = QtWidgets.QLabel()
        self.avatar_label.setFixedSize(28, 28)
        self.avatar_label.setScaledContents(True)
        left_col = QtWidgets.QWidget()
        left_col.setFixedWidth(36)
        left_col_layout = QtWidgets.QVBoxLayout(left_col)
        left_col_layout.setContentsMargins(4, 0, 4, 0)
        left_col_layout.setSpacing(0)
        left_col_layout.addStretch(1)
        if role == "assistant":
            left_col_layout.addWidget(self.avatar_label, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        else:
            self.avatar_label.hide()
        left_col_layout.addStretch(1)

        # Content block
        content_wrap = QtWidgets.QVBoxLayout()
        content_wrap.setContentsMargins(0, 0, 0, 0)
        content_wrap.setSpacing(4)
        self._content_wrap = content_wrap

        # Header with title, timestamp, and optional stats
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_left = QtWidgets.QHBoxLayout()
        header_left.setSpacing(6)
        self.header = QtWidgets.QLabel(f"{title} · {self._created_at.strftime('%H:%M')}")
        self.header.setObjectName("msgHeader")
        self.header.setProperty("meta", True)
        header_left.addWidget(self.header)
        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setObjectName("msgHeader")
        self.stats_label.setProperty("meta", True)
        self.stats_label.hide()
        header_left.addWidget(self.stats_label)
        header_layout.addLayout(header_left)
        header_layout.addStretch(1)
        self.copy_btn = QtWidgets.QToolButton()
        self.copy_btn.setObjectName("copyBtn")
        if qta:
            self.copy_btn.setIcon(qta.icon("mdi.content-copy"))
        else:
            self.copy_btn.setText("Copy")
        self.copy_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        header_layout.addWidget(self.copy_btn)
        content_wrap.addLayout(header_layout)

        # Bubble frame containing a QTextBrowser
        self.bubble = QtWidgets.QFrame()
        self.bubble.setObjectName("chatBubble")
        self.bubble.setProperty("user", "true" if role == "user" else "false")
        bubble_layout = QtWidgets.QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)
        bubble_layout.setSpacing(0)
        self.text = AutoResizeTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.setAcceptRichText(True)
        bubble_layout.addWidget(self.text)
        content_wrap.addWidget(self.bubble)
        self.bubble.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)

        # Arrange aligned full-width layout: fixed left avatar column + expanding content
        if role == "assistant":
            self._set_avatar_pixmap(_load_pixmap(config.ASSISTANT_AVATAR_PATH))
        root.addWidget(left_col)
        root.addLayout(content_wrap, 1)

    def _set_avatar_pixmap(self, pm: QtGui.QPixmap) -> None:
        self.avatar_label.setPixmap(pm)

    def set_plain_text(self, content: str) -> None:
        self._plain_accumulator = content
        html_content = f"<div style='white-space:pre-wrap'>{html.escape(content)}</div>"
        self.text.setHtml(self._wrap_html(html_content))

    def set_markdown(self, content: str) -> None:
        self._plain_accumulator = content
        self.text.setHtml(self._wrap_html(self._render_markdown(content)))

    def append_stream_text(self, text: str) -> None:
        self._plain_accumulator += text
        self.text.setHtml(self._wrap_html(self._render_markdown(self._plain_accumulator)))
        self._ensure_parent_scroll_to_bottom()

    def _copy_to_clipboard(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self._plain_accumulator)

    @staticmethod
    def _wrap_html(inner: str) -> str:
        style = """
        <style>
          body { color: #e6edf3; }
          h1,h2,h3,h4 { color:#e6edf3; margin:12px 0 6px; }
          p,li { line-height: 1.5; }
          code { background:#0b1020; color:#e6edf3; padding:2px 4px; border-radius:4px; }
          pre { background:#0b1020; color:#e6edf3; padding:10px; border-radius:8px; overflow:auto; }
          pre.math, code.math { background:#0b1020; color:#e6edf3; }
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #30363d; padding: 6px 8px; }
          blockquote { border-left:3px solid #30363d; padding-left:8px; color:#adb6c2; }
          a { color: #58a6ff; }
        </style>
        """
        return f"<html><head>{style}</head><body>{inner}</body></html>"

    @staticmethod
    def _render_markdown(content: str) -> str:
        try:
            import markdown2
            from html import unescape as html_unescape
            from pygments import highlight  # type: ignore
            from pygments.lexers import get_lexer_by_name, guess_lexer  # type: ignore
            from pygments.formatters import HtmlFormatter  # type: ignore
            # Protect LaTeX-style math so markdown doesn't mangle it; render as styled code
            inline_pat = re.compile(r"\\\((.+?)\\\)")
            block_pat_dollar = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
            block_pat_bracket = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)

            tokens: list[str] = []

            def _store(token_text: str, block: bool) -> str:
                idx = len(tokens)
                tokens.append((token_text, block))
                return f"@@MATH{idx}@@"

            def repl_inline(m: re.Match) -> str:
                return _store(m.group(1), False)

            def repl_block(m: re.Match) -> str:
                return _store(m.group(1), True)

            protected = block_pat_dollar.sub(repl_block, content)
            protected = block_pat_bracket.sub(repl_block, protected)
            protected = inline_pat.sub(repl_inline, protected)

            html_out = markdown2.markdown(
                protected,
                extras=["fenced-code-blocks", "tables", "strike", "task_list", "code-friendly"],
            )  # type: ignore

            # Syntax highlight fenced code blocks
            code_block_re = re.compile(r"<pre><code(?: class=\"language-([a-zA-Z0-9_\-]+)\")?>([\s\S]+?)</code></pre>")
            def _hl(match: re.Match) -> str:
                lang = match.group(1)
                raw_code = html_unescape(match.group(2))
                try:
                    lexer = get_lexer_by_name(lang) if lang else guess_lexer(raw_code)
                except Exception:
                    from pygments.lexers import TextLexer  # type: ignore
                    lexer = TextLexer()
                formatter = HtmlFormatter(noclasses=True)
                highlighted = highlight(raw_code, lexer, formatter)
                return f"<pre>{highlighted}</pre>"
            html_out = code_block_re.sub(_hl, html_out)

            # Restore math tokens as styled blocks
            for i, (tex_src, is_block) in enumerate(tokens):
                safe = html.escape(tex_src)
                if is_block:
                    replacement = f"<pre class='math'>$$ {safe} $$</pre>"
                else:
                    replacement = f"<code class='math'>\\({safe}\\)</code>"
                html_out = html_out.replace(f"@@MATH{i}@@", replacement)

            return html_out
        except Exception:
            return f"<pre>{html.escape(content)}</pre>"

    # --- Reasoning ---
    def ensure_reasoning_controls(self) -> None:
        if hasattr(self, "reasoning_container"):
            return
        self.reasoning_container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.reasoning_container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        header = QtWidgets.QHBoxLayout()
        self.reasoning_label = QtWidgets.QLabel("Reasoning")
        self.reasoning_label.setStyleSheet("color:#8b949e; font-size:12px;")
        header.addWidget(self.reasoning_label)
        header.addStretch(1)
        self.reasoning_toggle = QtWidgets.QToolButton()
        self.reasoning_toggle.setText("Show")
        self.reasoning_toggle.setCheckable(True)
        self.reasoning_toggle.setChecked(False)
        self.reasoning_toggle.clicked.connect(self._toggle_reasoning)
        header.addWidget(self.reasoning_toggle)
        layout.addLayout(header)
        self.reasoning_view = AutoResizeTextBrowser()
        self.reasoning_view.setHtml(self._wrap_html(""))
        self.reasoning_view.setVisible(False)
        layout.addWidget(self.reasoning_view)
        # Insert reasoning block ABOVE the assistant bubble, right after the row header
        # Prefer modern layout; gracefully handle legacy attribute if present
        target_layout = getattr(self, "_content_wrap", None)
        if target_layout is None and hasattr(self, "_outer_layout"):
            target_layout = getattr(self, "_outer_layout")
        if isinstance(target_layout, (QtWidgets.QBoxLayout,)):
            target_layout.insertWidget(1, self.reasoning_container)
        else:
            # Fallback to top-level layout
            top_layout = self.layout()
            if isinstance(top_layout, (QtWidgets.QBoxLayout,)):
                top_layout.insertWidget(0, self.reasoning_container)
            else:
                # As a last resort, just show below header
                self._content_wrap.addWidget(self.reasoning_container)

    def _toggle_reasoning(self) -> None:
        visible = self.reasoning_toggle.isChecked()
        self.reasoning_toggle.setText("Hide" if visible else "Show")
        self.reasoning_view.setVisible(visible)

    def append_reasoning(self, text: str) -> None:
        self.ensure_reasoning_controls()
        # Keep as plain preformatted text to avoid markdown confusion
        current = getattr(self, "_reasoning_accumulator", "") + text
        self._reasoning_accumulator = current
        html_block = f"<pre style='white-space:pre-wrap'>{html.escape(current)}</pre>"
        self.reasoning_view.setHtml(self._wrap_html(html_block))
        self._ensure_parent_scroll_to_bottom()
        # Auto-show reasoning when content arrives
        if not self.reasoning_toggle.isChecked():
            self.reasoning_toggle.setChecked(True)
            self._toggle_reasoning()
            
    def append_stats(self, stats: str) -> None:
        """Add performance stats to the message header."""
        self.stats_label.setText(f"· {stats}")
        self.stats_label.show()

    def _ensure_parent_scroll_to_bottom(self) -> None:
        # Ask parent window to scroll if user is at bottom
        w = self.parent()
        while w and not isinstance(w, QtWidgets.QMainWindow):
            w = w.parent()
        if w and hasattr(w, "scroll_to_bottom_if_needed"):
            getattr(w, "scroll_to_bottom_if_needed")()


class StreamWorker(QtCore.QObject):
    token = QtCore.Signal(str)
    thinking = QtCore.Signal(str)
    finished = QtCore.Signal(str, str, float, float, int)  # content, thinking, reasoning_time, response_time, token_count

    def __init__(self, model_name: str, messages, options) -> None:
        super().__init__()
        self._model_name = model_name
        self._messages = messages
        self._options = options
        self._token_count = 0
        self._thinking_start = None
        self._response_start = None
        self._thinking_time = 0.0
        self._response_time = 0.0

    @QtCore.Slot()
    def run(self) -> None:
        from time import time
        full_response: list[str] = []
        full_thinking: list[str] = []
        
        self._thinking_start = time()
        self._response_start = None
        
        for chunk in ollama_chat(
            model=self._model_name,
            messages=self._messages,
            stream=True,
            options=self._options,
        ):
            msg = chunk.get("message", {})
            if (thinking := msg.get("thinking")):
                # Still in thinking phase
                full_thinking.append(thinking)
                self.thinking.emit(thinking)
            if (content := msg.get("content")):
                # First content token marks end of thinking, start of response
                if self._response_start is None:
                    self._thinking_time = time() - self._thinking_start
                    self._response_start = time()
                full_response.append(content)
                self._token_count += 1  # Rough approximation: each chunk is ~1 token
                self.token.emit(content)
        
        # Calculate final timings
        if self._response_start:
            self._response_time = time() - self._response_start
        else:
            # No response phase (error?)
            self._response_time = 0
            self._thinking_time = time() - self._thinking_start
            
        self.finished.emit(
            "".join(full_response),
            "".join(full_thinking),
            self._thinking_time,
            self._response_time,
            self._token_count
        )

class ToastOverlay(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("toast")
        self.setProperty("class", "toast")
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self.label = QtWidgets.QLabel("")
        layout.addWidget(self.label)
        self._opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._anim = QtCore.QPropertyAnimation(self._opacity, b"opacity", self)
        self._anim.setDuration(200)
        self.hide()

    def show_message(self, text: str, duration_ms: int = 1600) -> None:
        self.label.setText(text)
        self.adjustSize()
        parent = self.parentWidget()
        if not parent:
            return
        geo = parent.geometry()
        self.move(geo.width() - self.width() - 24, geo.height() - self.height() - 24)
        self._opacity.setOpacity(0.0)
        self.show()
        self._anim.stop()
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()
        QtCore.QTimer.singleShot(duration_ms, self._fade_out)

    def _fade_out(self) -> None:
        self._anim.stop()
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.start()
        self._anim.finished.connect(self.hide)


class _NoDragHandle(QtWidgets.QSplitterHandle):
    def __init__(self, orientation: QtCore.Qt.Orientation, parent: QtWidgets.QSplitter) -> None:
        super().__init__(orientation, parent)
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.setFixedWidth(0)

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return QtCore.QSize(0, 0)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        # Do not paint any handle visuals
        return

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        event.ignore()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        event.ignore()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        event.ignore()


class FixedSidebarSplitter(QtWidgets.QSplitter):
    def createHandle(self) -> QtWidgets.QSplitterHandle:  # type: ignore[override]
        return _NoDragHandle(self.orientation(), self)

class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GPT-OSS Desktop")
        self.resize(1000, 760)
        try:
            self.setWindowIcon(QtGui.QIcon("assets/openai.png"))
        except Exception:
            pass

        # --- State ---
        self.selected_prompt_index = 0
        self.selected_reasoning_index = 0
        self.add_special_message = False

        self.session = ChatSession(
            base_system_prompt=SYSTEM_PROMPTS[self.selected_prompt_index]["prompt"],
            model_name=config.MODEL_NAME,
        )

        # --- UI ---
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # No global header bar per design

        # Splitter: left (settings) / right (chat)
        splitter = FixedSidebarSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter = splitter
        main_layout.addWidget(splitter, 1)

        # Settings panel (modernized sidebar)
        sidebar_container = QtWidgets.QScrollArea()
        sidebar_container.setObjectName("sidebarScroll")
        sidebar_container.setWidgetResizable(True)
        sidebar_container.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_container.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # Use zero viewport margins; spacing is handled by inner layouts
        sidebar_container.setViewportMargins(0, 0, 0, 0)
        sidebar = QtWidgets.QWidget()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(16)

        card = QtWidgets.QFrame()
        card.setObjectName("sidebarCard")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(16)

        # Sidebar header
        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        if qta:
            icon_label = QtWidgets.QLabel()
            icon_label.setPixmap(qta.icon("mdi.tune").pixmap(16, 16))
            header_row.addWidget(icon_label)
        title_label = QtWidgets.QLabel("Settings")
        title_label.setObjectName("sidebarTitle")
        header_row.addWidget(title_label)
        header_row.addStretch(1)
        # Collapser in top-right of sidebar
        self.sidebar_collapse_btn = QtWidgets.QToolButton()
        if qta:
            self.sidebar_collapse_btn.setIcon(qta.icon("mdi.chevron-left"))
        else:
            self.sidebar_collapse_btn.setText("<")
        self.sidebar_collapse_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sidebar_collapse_btn.clicked.connect(self._toggle_settings)
        header_row.addWidget(self.sidebar_collapse_btn)
        card_layout.addLayout(header_row)

        self.compliance_checkbox = QtWidgets.QCheckBox("Compliance Protocol")
        self.compliance_checkbox.stateChanged.connect(self._on_toggle_compliance)
        card_layout.addWidget(self.compliance_checkbox)

        form = QtWidgets.QFormLayout()
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        prompt_names = [p["name"] for p in SYSTEM_PROMPTS]
        self.prompt_combo = QtWidgets.QComboBox()
        self.prompt_combo.addItems(prompt_names)
        self.prompt_combo.currentIndexChanged.connect(self._on_prompt_changed)
        self._configure_combo(self.prompt_combo)
        form.addRow("System Prompt", self.prompt_combo)

        self.reasoning_combo = QtWidgets.QComboBox()
        self.reasoning_combo.addItems(REASONING_OPTIONS)
        self.reasoning_combo.currentIndexChanged.connect(self._on_reasoning_changed)
        self._configure_combo(self.reasoning_combo)
        form.addRow("Reasoning Effort", self.reasoning_combo)

        card_layout.addLayout(form)
        sidebar_layout.addWidget(card)
        sidebar_layout.addStretch(1)

        sidebar_container.setWidget(sidebar)
        splitter.addWidget(sidebar_container)
        # Widen to ensure no clipping of content even with right padding
        sidebar_container.setMinimumWidth(380)
        sidebar_container.setMaximumWidth(380)

        # Chat panel
        chat_panel = QtWidgets.QWidget()
        chat_layout = QtWidgets.QVBoxLayout(chat_panel)

        self.chat_scroll = QtWidgets.QScrollArea()
        self.chat_scroll.setObjectName("chatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_canvas = QtWidgets.QWidget()
        self.chat_canvas.setObjectName("chatCanvas")
        self.messages_layout = QtWidgets.QVBoxLayout(self.chat_canvas)
        self.messages_layout.setContentsMargins(24, 12, 24, 12)
        self.messages_layout.setSpacing(0)
        self.messages_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_canvas)
        chat_layout.addWidget(self.chat_scroll, 1)

        # Input area like ChatGPT (multi-line, growing)
        input_container = QtWidgets.QFrame()
        input_container.setObjectName("inputContainer")
        input_layout = QtWidgets.QHBoxLayout(input_container)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(8)
        self.input_edit = GrowingPlainTextEdit(min_height=44, max_height=140)
        self.input_edit.setObjectName("chatInput")
        self.input_edit.setPlaceholderText("Ask...")
        self.input_edit.sendRequested.connect(self._on_send)
        input_layout.addWidget(self.input_edit, 1)
        # Extra shortcut binding on the input for reliability
        QtGui.QShortcut(QtGui.QKeySequence("Meta+Return"), self.input_edit, activated=self._on_send)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self.input_edit, activated=self._on_send)
        # Ensure correct initial compact height
        self.input_edit.adjustHeight()
        self.send_button = QtWidgets.QPushButton()
        if qta:
            self.send_button.setIcon(qta.icon("mdi.send"))
        else:
            self.send_button.setText("Send")
        self.send_button.clicked.connect(self._on_send)
        input_layout.addWidget(self.send_button)
        chat_layout.addWidget(input_container)

        splitter.addWidget(chat_panel)
        splitter.setStretchFactor(1, 1)

        # Render any prior messages (none at start besides system)
        self._render_history()

        # Keyboard shortcut: Enter to send, Shift+Enter for newline
        # Keyboard shortcut: Cmd+Enter to send
        QtGui.QShortcut(QtGui.QKeySequence("Meta+Return"), self, activated=self._on_send)

        # Track whether to auto-scroll as content streams
        self._auto_scroll_enabled = True
        self.chat_scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)

        # Toast overlay
        self._toast = ToastOverlay(self)

        # Overlay expand button visibility is controlled manually on toggle

        # Expand button overlay (visible when sidebar hidden)
        self._expand_sidebar_btn = QtWidgets.QToolButton(self)
        if qta:
            self._expand_sidebar_btn.setIcon(qta.icon("mdi.chevron-right"))
        else:
            self._expand_sidebar_btn.setText(">")
        self._expand_sidebar_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self._expand_sidebar_btn.setStyleSheet(
            "QToolButton { border: 1px solid #242b36; border-radius: 12px; padding: 2px; background: rgba(255,255,255,0.04); }"
        )
        self._expand_sidebar_btn.setFixedSize(24, 24)
        self._expand_sidebar_btn.setIconSize(QtCore.QSize(14, 14))
        self._expand_sidebar_btn.clicked.connect(self._toggle_settings)
        self._expand_sidebar_btn.hide()
        self._position_expand_button()
        # After first show/layout pass, re-sync visibility in case early resize
        # events ran before child widgets reported correct visibility
        QtCore.QTimer.singleShot(0, self._apply_responsive_sidebar)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        # Ensure the expand button is hidden when the sidebar is visible on first show
        self._apply_responsive_sidebar()

    def _configure_combo(self, combo: QtWidgets.QComboBox) -> None:
        combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        combo.setMinimumHeight(32)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        view = QtWidgets.QListView()
        view.setUniformItemSizes(False)
        view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        view.setSpacing(2)
        view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        combo.setView(view)
        # Widen popup to fit longest item
        fm = combo.fontMetrics()
        try:
            max_w = max((fm.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())), default=160)
        except Exception:
            max_w = 160
        view.setMinimumWidth(max(180, max_w + 36))

    # --- Slots ---
    def _on_toggle_compliance(self, state: int) -> None:
        # Debug the state change
        print(f"\n=== Compliance Toggle ===")
        print(f"State value: {state}")
        print(f"Qt.Checked value: {QtCore.Qt.CheckState.Checked}")
        self.add_special_message = bool(state)  # Convert to boolean - any non-zero state means checked
        print(f"add_special_message set to: {self.add_special_message}")
        print("=== End Toggle ===\n")

    def _on_prompt_changed(self, index: int) -> None:
        self.selected_prompt_index = index
        self.session.base_system_prompt = SYSTEM_PROMPTS[index]["prompt"]
        self.session.reset_messages()
        self._render_history()

    def _on_reasoning_changed(self, index: int) -> None:
        self.selected_reasoning_index = index

    def _append_chat(self, role: str, content: str, thinking: Optional[str] = None) -> None:
        if role == "user":
            row = MessageRow(role="user", title="You")
            row.set_plain_text(content)
        else:
            row = MessageRow(role="assistant", title="Assistant")
            row.set_markdown(content)
            if thinking:
                row.append_reasoning(thinking)
        # Connect copy toast
        row.copy_btn.clicked.connect(lambda: self._toast.show_message("Copied"))
        self._add_row(row)

    def _render_history(self) -> None:
        # Clear messages area (rebuild canvas)
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for m in self.session.messages:
            if m.role == "user":
                self._append_chat("user", m.content)
            elif m.role == "assistant":
                self._append_chat("assistant", m.content, m.thinking)

    def _on_send(self) -> None:
        user_text = self.input_edit.toPlainText().strip()
        if not user_text:
            return
        self.input_edit.clear()

        # Display user message immediately
        self.session.add_user_message(user_text)
        self._append_chat("user", user_text)

        self._start_stream_thread(user_text)
    def _start_stream_thread(self, user_text: str) -> None:
        reasoning_effort = REASONING_OPTIONS[self.selected_reasoning_index]

        model_messages, model_options, message_to_send = self.session.build_stream_payload(
            user_input=user_text,
            add_special_message=self.add_special_message,
            reasoning_effort=reasoning_effort,
        )
        
        # Debug print what's being sent
        print("\n=== Message sent to model ===")
        print(f"Compliance: {self.add_special_message}")
        print(f"Original text: {user_text}")
        print(f"Transformed message: {message_to_send}")
        print("=== End debug ===\n")

        # Prepare UI: add an Assistant bubble and stream text into it
        self._current_assistant_bubble = MessageRow(role="assistant", title="Assistant")
        self._current_assistant_bubble.set_plain_text("")
        self._add_row(self._current_assistant_bubble)

        self._stream_thread = QtCore.QThread(self)  # keep reference
        self._worker = StreamWorker(self.session.model_name, model_messages, model_options)
        self._worker.moveToThread(self._stream_thread)

        self._stream_thread.started.connect(self._worker.run)
        self._worker.token.connect(self._on_token)
        self._worker.thinking.connect(self._on_thinking)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._stream_thread.finished.connect(self._stream_thread.deleteLater)
        self._stream_thread.start()

    @QtCore.Slot(str)
    def _on_token(self, text: str) -> None:
        if hasattr(self, "_current_assistant_bubble") and self._current_assistant_bubble:
            self._current_assistant_bubble.append_stream_text(text)
            self.scroll_to_bottom_if_needed()

    @QtCore.Slot(str)
    def _on_thinking(self, text: str) -> None:
        # Append streamed reasoning into the current assistant row
        if hasattr(self, "_current_assistant_bubble") and self._current_assistant_bubble:
            self._current_assistant_bubble.append_reasoning(text)
            self.scroll_to_bottom_if_needed()

    @QtCore.Slot(str, str)
    def _on_finished(self, content: str, thinking: str, reasoning_time: float, response_time: float, token_count: int) -> None:
        if hasattr(self, "_current_assistant_bubble") and self._current_assistant_bubble:
            self._current_assistant_bubble.set_markdown(content)
            if thinking:
                # Ensure reasoning section exists but keep collapsed by default
                self._current_assistant_bubble.append_reasoning("")
            # Add performance stats
            tokens_per_sec = token_count / response_time if response_time > 0 else 0
            stats = f"{reasoning_time:.1f}s (reasoning) · {response_time:.1f}s (responding) · {tokens_per_sec:.1f} tokens/s"
            self._current_assistant_bubble.append_stats(stats)
            self._current_assistant_bubble = None
        self.session.add_assistant_message(content, thinking or None)
        if hasattr(self, "_stream_thread"):
            self._stream_thread.quit()
            self._stream_thread.wait()
        self.scroll_to_bottom_if_needed()

    # --- Message helpers ---
    def _add_row(self, row: 'MessageRow') -> None:
        # Insert above the stretch item
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, row)
        QtCore.QTimer.singleShot(0, lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))

    def _auto_resize_input(self) -> None:
        doc = self.input_edit.document()
        doc.setTextWidth(self.input_edit.viewport().width())
        height = min(140, max(44, int(doc.size().height()) + 10))
        self.input_edit.setFixedHeight(height)

    def _toggle_settings(self) -> None:
        # Show/hide the left pane
        splitter: QtWidgets.QSplitter = self.splitter
        left = splitter.widget(0)
        was_visible = left.isVisible()
        left.setVisible(not was_visible)
        if self._expand_sidebar_btn is None and was_visible:
            # Lazily create expand button when first needed
            self._expand_sidebar_btn = QtWidgets.QToolButton(self)
            if qta:
                self._expand_sidebar_btn.setIcon(qta.icon("mdi.chevron-right"))
            else:
                self._expand_sidebar_btn.setText(">")
            self._expand_sidebar_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self._expand_sidebar_btn.setStyleSheet(
                "QToolButton { border: 1px solid #242b36; border-radius: 12px; padding: 2px; background: rgba(255,255,255,0.04); }"
            )
            self._expand_sidebar_btn.setFixedSize(24, 24)
            self._expand_sidebar_btn.setIconSize(QtCore.QSize(14, 14))
            self._expand_sidebar_btn.clicked.connect(self._toggle_settings)
        if self._expand_sidebar_btn:
            self._expand_sidebar_btn.setVisible(not left.isVisible())
            self._position_expand_button()

    def _new_chat(self) -> None:
        self.session.reset_messages()
        self._render_history()

    # --- Scrolling helpers ---
    def _on_scroll(self) -> None:
        sb = self.chat_scroll.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 2
        self._auto_scroll_enabled = at_bottom

    def scroll_to_bottom_if_needed(self) -> None:
        if self._auto_scroll_enabled:
            QtCore.QTimer.singleShot(0, lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))

    # --- Responsive helpers ---
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_responsive_sidebar()
        if self._expand_sidebar_btn:
            self._position_expand_button()

    def _apply_responsive_sidebar(self) -> None:
        # Keep behavior simple: do not auto-hide to avoid flicker; rely on user toggle
        try:
            sidebar = self.splitter.widget(0)
            if self._expand_sidebar_btn:
                self._expand_sidebar_btn.setVisible(not sidebar.isVisible())
        except Exception:
            pass

    def _position_expand_button(self) -> None:
        # Place near top-left edge of chat panel when sidebar hidden
        if not self._expand_sidebar_btn:
            return
        geo = self.geometry()
        x = 12
        y = 12
        self._expand_sidebar_btn.move(x, y)


def run() -> None:
    setup_pre_qapp()
    app = QtWidgets.QApplication(sys.argv)
    setup_hidpi_and_font(app)
    ThemeManager().apply(app)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()


