from __future__ import annotations

import sys
import html
from typing import Optional, List, Dict
import re
from datetime import datetime
import math

from PySide6 import QtCore, QtGui, QtWidgets
from ollama import chat as ollama_chat

import config
from system_prompts import SYSTEM_PROMPTS
from core.chat_service import ChatSession, ChatMessage, REASONING_OPTIONS
from core.m2m_formatter import parse_m2m_output, format_m2m_to_markdown, is_m2m_format, debug_print_parsed_data
from core.chat_persistence import ChatPersistence
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
        # Handle paste events
        if e.matches(QtGui.QKeySequence.StandardKey.Paste):
            # Let the paste event handler in the parent window handle image paste
            parent = self.parent()
            while parent and not isinstance(parent, QtWidgets.QMainWindow):
                parent = parent.parent()
            if parent and hasattr(parent, '_handle_paste'):
                if parent._handle_paste():
                    e.accept()
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
        self._apply_m2m_formatting: bool = False
        self._m2m_debug_shown: bool = False

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

    def set_markdown(self, content: str, apply_m2m_formatting: bool = False) -> None:
        self._plain_accumulator = content
        
        # Apply M2M formatting if requested
        if apply_m2m_formatting and is_m2m_format(content):
            # Debug output for M2M
            print("\n=== M2M Raw Output (Final) ===")
            print(content)
            print("=== End M2M Raw Output ===")
            
            parsed_data = parse_m2m_output(content)
            if parsed_data:
                debug_print_parsed_data(parsed_data)
                formatted_content = format_m2m_to_markdown(parsed_data)
                self.text.setHtml(self._wrap_html(self._render_markdown(formatted_content)))
                return
        
        # Regular markdown rendering
        self.text.setHtml(self._wrap_html(self._render_markdown(content)))

    def enable_m2m_formatting(self) -> None:
        """Enable M2M formatting for this message."""
        self._apply_m2m_formatting = True

    def append_stream_text(self, text: str) -> None:
        self._plain_accumulator += text
        
        # Apply M2M formatting if enabled
        if self._apply_m2m_formatting and is_m2m_format(self._plain_accumulator):
            # Show debug output only once when we first detect M2M format
            if not self._m2m_debug_shown:
                print("\n=== M2M Raw Output (Streaming) ===")
                print(self._plain_accumulator)
                print("... (streaming continues)")
                print("=== End M2M Raw Output ===\n")
                self._m2m_debug_shown = True
            
            parsed_data = parse_m2m_output(self._plain_accumulator)
            if parsed_data:
                formatted_content = format_m2m_to_markdown(parsed_data)
                self.text.setHtml(self._wrap_html(self._render_markdown(formatted_content)))
                self._ensure_parent_scroll_to_bottom()
                return
        
        # Regular markdown rendering
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
    
    def add_images(self, images: List[Dict[str, str]]) -> None:
        """Add images to the message content."""
        if not images:
            return
        
        # Create a container for images
        image_container = QtWidgets.QWidget()
        image_layout = QtWidgets.QHBoxLayout(image_container)
        image_layout.setContentsMargins(0, 8, 0, 8)
        image_layout.setSpacing(8)
        
        for image_info in images:
            # Create image widget
            image_label = QtWidgets.QLabel()
            image_label.setMaximumSize(200, 200)
            image_label.setScaledContents(False)
            image_label.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            
            # Load and display the image
            pixmap = QtGui.QPixmap()
            pixmap.loadFromData(QtCore.QByteArray.fromBase64(image_info["data"].encode()))
            if not pixmap.isNull():
                # Scale to fit while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(200, 200, QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
                
                # Store original pixmap for full view
                image_label.setProperty("original_pixmap", pixmap)
                
                # Add click handler to view full size
                image_label.mousePressEvent = lambda e, p=pixmap: self._show_full_image(p)
            
            image_layout.addWidget(image_label)
        
        image_layout.addStretch()
        
        # Insert the image container after the bubble
        self._content_wrap.insertWidget(2, image_container)  # Index 2 to place after header and bubble
    
    def _show_full_image(self, pixmap: QtGui.QPixmap) -> None:
        """Show full size image in a dialog."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Image")
        dialog.setModal(True)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scrollable image viewer
        scroll = QtWidgets.QScrollArea()
        image_label = QtWidgets.QLabel()
        image_label.setPixmap(pixmap)
        scroll.setWidget(image_label)
        
        layout.addWidget(scroll)
        
        # Set dialog size to 80% of screen or image size, whichever is smaller
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        max_width = int(screen.width() * 0.8)
        max_height = int(screen.height() * 0.8)
        
        dialog_width = min(pixmap.width() + 20, max_width)
        dialog_height = min(pixmap.height() + 20, max_height)
        dialog.resize(dialog_width, dialog_height)
        
        dialog.exec()

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
    error = QtCore.Signal(str)  # Error message

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
        self._stop_requested = False
    
    def request_stop(self) -> None:
        """Request the streaming to stop."""
        self._stop_requested = True

    @QtCore.Slot()
    def run(self) -> None:
        from time import time
        full_response: list[str] = []
        full_thinking: list[str] = []
        
        self._thinking_start = time()
        self._response_start = None
        
        try:
            for chunk in ollama_chat(
                model=self._model_name,
                messages=self._messages,
                stream=True,
                options=self._options,
            ):
                # Check if stop was requested
                if self._stop_requested:
                    full_response.append("\n\n[Generation stopped by user]")
                    break
                    
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
        except Exception as e:
            # Handle various error types
            error_msg = str(e)
            if "failed to connect" in error_msg.lower() or "connection error" in error_msg.lower():
                self.error.emit("Failed to connect to Ollama. Please ensure Ollama is running.")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                self.error.emit(f"Model '{self._model_name}' not found. Please pull the model first.")
            elif "context length exceeded" in error_msg.lower():
                self.error.emit("Context length exceeded. Please start a new conversation.")
            else:
                self.error.emit(f"An error occurred: {error_msg}")
            
            # Emit empty finished signal to clean up UI state
            self.finished.emit("", "", 0, 0, 0)

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

class ImagePreviewWidget(QtWidgets.QFrame):
    """Image preview widget with hover-based remove button."""
    
    removeRequested = QtCore.Signal(int)
    imageClicked = QtCore.Signal(object)  # Emits the QPixmap
    
    def __init__(self, index: int, image_info: Dict[str, str]) -> None:
        super().__init__()
        self.index = index
        self.setObjectName("imagePreview")
        self.setFixedSize(80, 80)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Image label
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(80, 80)
        self.image_label.setScaledContents(False)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background: transparent; }")
        
        # Load and display the image
        self.original_pixmap = QtGui.QPixmap()
        self.original_pixmap.loadFromData(QtCore.QByteArray.fromBase64(image_info["data"].encode()))
        if not self.original_pixmap.isNull():
            # Scale to fit while maintaining aspect ratio (leaving room for border)
            scaled_pixmap = self.original_pixmap.scaled(72, 72, QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                                         QtCore.Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            # Make image clickable for full view
            self.image_label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        
        # Remove button (hidden by default)
        self.remove_btn = QtWidgets.QPushButton("✕")
        self.remove_btn.setObjectName("imageRemoveBtn")
        self.remove_btn.setFixedSize(24, 24)
        self.remove_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.remove_btn.clicked.connect(lambda: self.removeRequested.emit(self.index))
        
        layout.addWidget(self.image_label)
        
        # Position remove button in top-right corner (after adding image to layout)
        self.remove_btn.setParent(self)
        self.remove_btn.move(58, -2)  # Slightly overlap the corner
        self.remove_btn.raise_()
        self.remove_btn.hide()  # Hidden by default
    
    def enterEvent(self, event: QtCore.QEvent) -> None:
        """Show remove button on hover."""
        self.remove_btn.show()
        self.remove_btn.raise_()  # Ensure it's on top
        super().enterEvent(event)
    
    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Hide remove button when not hovering."""
        self.remove_btn.hide()
        super().leaveEvent(event)
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle clicks on the image to show full view."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Check if click is on the remove button first
            if self.remove_btn.isVisible() and self.remove_btn.geometry().contains(event.pos()):
                # Let the remove button handle the click
                super().mousePressEvent(event)
                return
            
            # Check if click is on the image area
            image_rect = self.image_label.geometry()
            if image_rect.contains(event.pos()):
                self.imageClicked.emit(self.original_pixmap)
                event.accept()
                return
        super().mousePressEvent(event)


class ChatHistoryItem(QtWidgets.QWidget):
    """A clickable chat history item in the sidebar."""
    
    clicked = QtCore.Signal(str)  # Emits chat ID when clicked
    deleteRequested = QtCore.Signal(str)  # Emits chat ID when delete requested
    
    def __init__(self, chat_id: str, title: str, timestamp: str, message_count: int) -> None:
        super().__init__()
        self.chat_id = chat_id
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setObjectName("chatHistoryItem")
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        
        # Text info
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(2)
        
        # Title
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("chatHistoryTitle")
        self.title_label.setWordWrap(True)
        text_layout.addWidget(self.title_label)
        
        # Metadata
        meta_text = f"{timestamp} · {message_count} messages"
        self.meta_label = QtWidgets.QLabel(meta_text)
        self.meta_label.setObjectName("chatHistoryMeta")
        text_layout.addWidget(self.meta_label)
        
        layout.addLayout(text_layout, 1)
        
        # Delete button (hidden by default)
        self.delete_btn = QtWidgets.QToolButton()
        if qta:
            self.delete_btn.setIcon(qta.icon("mdi.delete-outline"))
        else:
            self.delete_btn.setText("X")
        self.delete_btn.setObjectName("chatHistoryDelete")
        self.delete_btn.hide()
        self.delete_btn.clicked.connect(lambda: self.deleteRequested.emit(self.chat_id))
        layout.addWidget(self.delete_btn)
        
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self.chat_id)
    
    def enterEvent(self, event: QtCore.QEvent) -> None:
        self.delete_btn.show()
        super().enterEvent(event)
    
    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.delete_btn.hide()
        super().leaveEvent(event)


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
        self.selected_model_index = 0  # Default to first model (gpt-oss:20b)
        self.current_chat_id: Optional[str] = None  # Track current chat
        self.selected_images: List[Dict[str, str]] = []  # List of {"data": base64_str, "type": mime_type, "path": file_path}

        self.session = ChatSession(
            base_system_prompt=SYSTEM_PROMPTS[self.selected_prompt_index]["prompt"],
            model_name=config.AVAILABLE_MODELS[self.selected_model_index]["name"],
        )
        
        # Initialize chat persistence
        self.chat_persistence = ChatPersistence()

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
        # Collapser positioned to match expander
        self.sidebar_collapse_btn = QtWidgets.QToolButton(self)
        if qta:
            self.sidebar_collapse_btn.setIcon(qta.icon("mdi.chevron-left"))
        else:
            self.sidebar_collapse_btn.setText("<")
        self.sidebar_collapse_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sidebar_collapse_btn.setStyleSheet(
            "QToolButton { border: 1px solid #242b36; border-radius: 12px; padding: 2px; background: rgba(255,255,255,0.04); }"
        )
        self.sidebar_collapse_btn.setFixedSize(24, 24)
        self.sidebar_collapse_btn.clicked.connect(self._toggle_settings)
        # Position it at the same coordinates as the expander and set initial visibility
        self.sidebar_collapse_btn.move(12, 12)
        # Initially visible since sidebar starts visible
        self.sidebar_collapse_btn.setVisible(True)
        card_layout.addLayout(header_row)

        # Model picker at the top
        form = QtWidgets.QFormLayout()
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        # Model selector dropdown
        model_names = [m["display_name"] for m in config.AVAILABLE_MODELS]
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(model_names)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._configure_combo(self.model_combo)
        form.addRow("Model", self.model_combo)

        # Compliance checkbox (only visible for gpt-oss)
        self.compliance_checkbox = QtWidgets.QCheckBox("Compliance Protocol")
        self.compliance_checkbox.stateChanged.connect(self._on_toggle_compliance)
        form.addRow("", self.compliance_checkbox)

        prompt_names = [p["name"] for p in SYSTEM_PROMPTS]
        self.prompt_combo = QtWidgets.QComboBox()
        self.prompt_combo.addItems(prompt_names)
        self.prompt_combo.currentIndexChanged.connect(self._on_prompt_changed)
        self._configure_combo(self.prompt_combo)
        form.addRow("System Prompt", self.prompt_combo)

        # Reasoning combo (only visible for reasoning models)
        self.reasoning_combo = QtWidgets.QComboBox()
        self.reasoning_combo.addItems(REASONING_OPTIONS)
        self.reasoning_combo.currentIndexChanged.connect(self._on_reasoning_changed)
        self._configure_combo(self.reasoning_combo)
        self.reasoning_label = QtWidgets.QLabel("Reasoning Effort")
        form.addRow(self.reasoning_label, self.reasoning_combo)

        card_layout.addLayout(form)
        sidebar_layout.addWidget(card)
        
        # Chat history section
        history_card = QtWidgets.QFrame()
        history_card.setObjectName("sidebarCard")
        history_layout = QtWidgets.QVBoxLayout(history_card)
        history_layout.setContentsMargins(16, 16, 16, 16)
        history_layout.setSpacing(12)
        
        # History header with new chat button
        history_header = QtWidgets.QHBoxLayout()
        history_header.setContentsMargins(0, 0, 0, 0)
        history_header.setSpacing(8)
        if qta:
            history_icon = QtWidgets.QLabel()
            history_icon.setPixmap(qta.icon("mdi.history").pixmap(16, 16))
            history_header.addWidget(history_icon)
        history_title = QtWidgets.QLabel("Chat History")
        history_title.setObjectName("sidebarTitle")
        history_header.addWidget(history_title)
        history_header.addStretch(1)
        
        # New chat button
        self.new_chat_btn = QtWidgets.QToolButton()
        if qta:
            self.new_chat_btn.setIcon(qta.icon("mdi.plus"))
        else:
            self.new_chat_btn.setText("+")
        self.new_chat_btn.setObjectName("newChatBtn")
        self.new_chat_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.new_chat_btn.clicked.connect(self._new_chat)
        history_header.addWidget(self.new_chat_btn)
        
        history_layout.addLayout(history_header)
        
        # Chat list
        self.chat_list_scroll = QtWidgets.QScrollArea()
        self.chat_list_scroll.setObjectName("chatListScroll")
        self.chat_list_scroll.setWidgetResizable(True)
        self.chat_list_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        
        self.chat_list_widget = QtWidgets.QWidget()
        self.chat_list_layout = QtWidgets.QVBoxLayout(self.chat_list_widget)
        self.chat_list_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_list_layout.setSpacing(4)
        self.chat_list_layout.addStretch(1)
        
        self.chat_list_scroll.setWidget(self.chat_list_widget)
        history_layout.addWidget(self.chat_list_scroll, 1)
        
        sidebar_layout.addWidget(history_card, 1)  # Give it stretch factor to take remaining space
        
        sidebar_container.setWidget(sidebar)
        splitter.addWidget(sidebar_container)
        # Widen to ensure no clipping of content even with right padding
        sidebar_container.setMinimumWidth(380)
        sidebar_container.setMaximumWidth(380)
        
        # Load chat history
        self._refresh_chat_history()

        # Chat panel
        chat_panel = QtWidgets.QWidget()
        chat_panel.setObjectName("chatPanel")
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

        # Image preview container (hidden by default)
        self.image_preview_container = QtWidgets.QFrame()
        self.image_preview_container.setObjectName("imagePreviewContainer")
        self.image_preview_container.setMaximumHeight(120)
        self.image_preview_container.hide()
        
        self.image_preview_layout = QtWidgets.QHBoxLayout(self.image_preview_container)
        self.image_preview_layout.setContentsMargins(8, 8, 8, 0)
        self.image_preview_layout.setSpacing(8)
        
        # Add a scroll area for image previews
        self.image_preview_scroll = QtWidgets.QScrollArea()
        self.image_preview_scroll.setWidgetResizable(True)
        self.image_preview_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_preview_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_preview_scroll.setMaximumHeight(100)
        
        self.image_preview_widget = QtWidgets.QWidget()
        self.image_preview_inner_layout = QtWidgets.QHBoxLayout(self.image_preview_widget)
        self.image_preview_inner_layout.setContentsMargins(0, 0, 0, 0)
        self.image_preview_inner_layout.setSpacing(8)
        self.image_preview_inner_layout.addStretch()
        
        self.image_preview_scroll.setWidget(self.image_preview_widget)
        self.image_preview_layout.addWidget(self.image_preview_scroll)
        
        chat_layout.addWidget(self.image_preview_container)

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
        
        # Image upload button
        self.image_button = QtWidgets.QPushButton()
        if qta:
            self.image_button.setIcon(qta.icon("mdi.image-plus"))
        else:
            self.image_button.setText("📎")
        self.image_button.setToolTip("Attach images")
        self.image_button.clicked.connect(self._on_add_image)
        input_layout.addWidget(self.image_button)
        
        self.send_button = QtWidgets.QPushButton()
        if qta:
            self.send_button.setIcon(qta.icon("mdi.send"))
        else:
            self.send_button.setText("Send")
        self.send_button.clicked.connect(self._on_send)
        input_layout.addWidget(self.send_button)
        
        # Stop button (hidden by default)
        self.stop_button = QtWidgets.QPushButton()
        self.stop_button.setObjectName("stopButton")
        if qta:
            self.stop_button.setIcon(qta.icon("mdi.stop"))
        else:
            self.stop_button.setText("Stop")
        self.stop_button.clicked.connect(self._on_stop_generation)
        self.stop_button.hide()
        input_layout.addWidget(self.stop_button)
        
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

        # Export button (floating in top right)
        self._export_btn = QtWidgets.QToolButton(self)
        if qta:
            self._export_btn.setIcon(qta.icon("mdi.download-outline", scale_factor=0.8))
        else:
            self._export_btn.setText("↓")
        self._export_btn.setToolTip("Export Chat")
        self._export_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self._export_btn.setObjectName("exportBtn")
        self._export_btn.setFixedSize(32, 32)
        self._export_btn.clicked.connect(self._export_chat)
        self._position_export_button()

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
        # Start hidden since sidebar is initially visible
        self._expand_sidebar_btn.setVisible(False)
        self._expand_sidebar_btn.setIconSize(QtCore.QSize(14, 14))
        self._expand_sidebar_btn.clicked.connect(self._toggle_settings)
        self._expand_sidebar_btn.hide()
        self._position_expand_button()
        # After first show/layout pass, re-sync visibility in case early resize
        # events ran before child widgets reported correct visibility
        QtCore.QTimer.singleShot(0, self._apply_responsive_sidebar)
        
        # Set initial model UI state
        self._update_ui_for_model(self.selected_model_index)

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
        # Don't do anything if this is the same prompt
        if index == self.selected_prompt_index:
            return
            
        # Save current chat if it has messages before switching prompts
        if self.current_chat_id and len(self.session.messages) > 1:
            self._save_current_chat(update_timestamp=False)
        
        self.selected_prompt_index = index
        
        # Start a new chat with the new prompt
        self.current_chat_id = None
        self.session = ChatSession(
            base_system_prompt=SYSTEM_PROMPTS[index]["prompt"],
            model_name=config.AVAILABLE_MODELS[self.selected_model_index]["name"],
        )
        
        self._render_history()
        self._refresh_chat_history()

    def _on_reasoning_changed(self, index: int) -> None:
        self.selected_reasoning_index = index

    def _update_ui_for_model(self, model_index: int) -> None:
        """Update UI elements based on model capabilities."""
        model_info = config.AVAILABLE_MODELS[model_index]
        
        # Enable/disable compliance checkbox based on model capabilities
        self.compliance_checkbox.setEnabled(model_info["supports_compliance"])
        if not model_info["supports_compliance"]:
            self.compliance_checkbox.setChecked(False)
            self.add_special_message = False
        
        # Enable/disable reasoning dropdown based on model capabilities
        self.reasoning_combo.setEnabled(model_info["supports_reasoning"])
        self.reasoning_label.setEnabled(model_info["supports_reasoning"])
        
        # Enable/disable image button based on model capabilities
        self.image_button.setEnabled(model_info.get("supports_images", False))
        if not model_info.get("supports_images", False):
            # Clear any selected images when switching to a model that doesn't support them
            self._clear_selected_images()

    def _on_model_changed(self, index: int) -> None:
        # Don't do anything if this is the same model
        if index == self.selected_model_index:
            return
            
        # Save current chat if it has messages before switching models
        if self.current_chat_id and len(self.session.messages) > 1:
            self._save_current_chat(update_timestamp=False)
        
        self.selected_model_index = index
        model_info = config.AVAILABLE_MODELS[index]
        
        # Start a new chat with the new model
        self.current_chat_id = None
        self.session = ChatSession(
            base_system_prompt=SYSTEM_PROMPTS[self.selected_prompt_index]["prompt"],
            model_name=model_info["name"],
        )
        
        # Update UI for the new model
        self._update_ui_for_model(index)
        
        # Render the new empty chat
        self._render_history()
        self._refresh_chat_history()

    def _append_chat(self, role: str, content: str, thinking: Optional[str] = None, images: Optional[List[Dict[str, str]]] = None) -> None:
        if role == "user":
            row = MessageRow(role="user", title="You")
            row.set_plain_text(content)
            if images:
                row.add_images(images)
        else:
            row = MessageRow(role="assistant", title="Assistant")
            # Check if M2M system prompt is selected
            is_m2m = (self.selected_prompt_index < len(SYSTEM_PROMPTS) and 
                     SYSTEM_PROMPTS[self.selected_prompt_index]["name"] == "M2M")
            row.set_markdown(content, apply_m2m_formatting=is_m2m)
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
                self._append_chat("user", m.content, images=m.images if m.images else None)
            elif m.role == "assistant":
                self._append_chat("assistant", m.content, m.thinking)

    def _on_send(self) -> None:
        user_text = self.input_edit.toPlainText().strip()
        if not user_text and not self.selected_images:
            return
        # If no text but images exist, add a default prompt
        if not user_text and self.selected_images:
            user_text = "What's in this image?"
        self.input_edit.clear()

        # Display user message immediately with images if any
        self.session.add_user_message(user_text, self.selected_images)
        self._append_chat("user", user_text, images=self.selected_images)
        
        # Clear selected images after sending
        self._clear_selected_images()

        self._start_stream_thread(user_text)
    def _start_stream_thread(self, user_text: str) -> None:
        # Get current model info
        model_info = config.AVAILABLE_MODELS[self.selected_model_index]
        
        # Only use reasoning effort if the model supports it
        if model_info["supports_reasoning"]:
            reasoning_effort = REASONING_OPTIONS[self.selected_reasoning_index]
        else:
            # For non-reasoning models, pass empty string to avoid GPT-specific logic
            reasoning_effort = ""

        model_messages, model_options, message_to_send = self.session.build_stream_payload(
            user_input=user_text,
            add_special_message=self.add_special_message if model_info["supports_compliance"] else False,
            reasoning_effort=reasoning_effort,
        )
        


        # Prepare UI: add an Assistant bubble and stream text into it
        self._current_assistant_bubble = MessageRow(role="assistant", title="Assistant")
        self._current_assistant_bubble.set_plain_text("")
        
        # Enable M2M formatting if M2M system prompt is selected
        is_m2m = (self.selected_prompt_index < len(SYSTEM_PROMPTS) and 
                 SYSTEM_PROMPTS[self.selected_prompt_index]["name"] == "M2M")
        if is_m2m:
            self._current_assistant_bubble.enable_m2m_formatting()
        
        self._add_row(self._current_assistant_bubble)

        # Update UI state for streaming
        self.send_button.hide()
        self.stop_button.show()
        self.input_edit.setEnabled(False)

        self._stream_thread = QtCore.QThread(self)  # keep reference
        self._worker = StreamWorker(self.session.model_name, model_messages, model_options)
        self._worker.moveToThread(self._stream_thread)

        self._stream_thread.started.connect(self._worker.run)
        self._worker.token.connect(self._on_token)
        self._worker.thinking.connect(self._on_thinking)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_stream_error)
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
    
    @QtCore.Slot(str)
    def _on_stream_error(self, error_msg: str) -> None:
        """Handle streaming errors by showing error message and cleaning up."""
        # Remove the empty assistant bubble if it exists
        if hasattr(self, "_current_assistant_bubble") and self._current_assistant_bubble:
            self._current_assistant_bubble.deleteLater()
            self._current_assistant_bubble = None
        
        # Reset UI state
        self.stop_button.hide()
        self.send_button.show()
        self.input_edit.setEnabled(True)
        self.input_edit.setFocus()
        
        # Show error in a message box
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)
        
        # Clean up the thread
        if hasattr(self, "_stream_thread"):
            self._stream_thread.quit()
            self._stream_thread.wait()

    @QtCore.Slot(str, str, float, float, int)
    def _on_finished(self, content: str, thinking: str, reasoning_time: float, response_time: float, token_count: int) -> None:
        if hasattr(self, "_current_assistant_bubble") and self._current_assistant_bubble:
            # Check if M2M formatting should be applied
            is_m2m = (self.selected_prompt_index < len(SYSTEM_PROMPTS) and 
                     SYSTEM_PROMPTS[self.selected_prompt_index]["name"] == "M2M")
            self._current_assistant_bubble.set_markdown(content, apply_m2m_formatting=is_m2m)
            if thinking:
                # Ensure reasoning section exists but keep collapsed by default
                self._current_assistant_bubble.append_reasoning("")
            # Add performance stats
            if response_time > 0:
                tokens_per_sec = token_count / response_time if response_time > 0 else 0
                stats = f"{reasoning_time:.1f}s (reasoning) · {response_time:.1f}s (responding) · {tokens_per_sec:.1f} tokens/s"
                self._current_assistant_bubble.append_stats(stats)
            self._current_assistant_bubble = None
        
        # Only add to session if we got actual content
        if content:
            self.session.add_assistant_message(content, thinking or None)
        
        if hasattr(self, "_stream_thread"):
            self._stream_thread.quit()
            self._stream_thread.wait()
        self.scroll_to_bottom_if_needed()
        
        # Reset UI state
        self.stop_button.hide()
        self.send_button.show()
        self.input_edit.setEnabled(True)
        self.input_edit.setFocus()
        
        # Auto-save the chat after each message exchange
        self._save_current_chat()
        self._refresh_chat_history()
    
    def _on_stop_generation(self) -> None:
        """Stop the current generation."""
        if hasattr(self, "_worker") and self._worker:
            self._worker.request_stop()
            self.stop_button.setEnabled(False)  # Prevent multiple clicks

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
        is_visible = not left.isVisible()  # The new state we're switching to
        left.setVisible(is_visible)
        
        # Ensure expand button exists
        if self._expand_sidebar_btn is None:
            # Create expand button if needed
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
            self._position_expand_button()
        
        # Update button visibility based on new sidebar state
        if self._expand_sidebar_btn:
            self._expand_sidebar_btn.setVisible(not is_visible)  # Show expand when sidebar is hidden
        if hasattr(self, 'sidebar_collapse_btn'):
            self.sidebar_collapse_btn.setVisible(is_visible)  # Show collapse when sidebar is visible
            if is_visible:
                self.sidebar_collapse_btn.move(12, 12)

    def _new_chat(self) -> None:
        """Create a new chat session."""
        # Save current chat if it has messages (without updating timestamp)
        if self.current_chat_id and len(self.session.messages) > 1:
            self._save_current_chat(update_timestamp=False)
        
        # Reset for new chat
        self.current_chat_id = None
        self.session.reset_messages()
        self._render_history()
        self._refresh_chat_history()
    
    def _save_current_chat(self, update_timestamp: bool = True) -> None:
        """Save the current chat session."""
        if len(self.session.messages) <= 1:  # Only system message
            return
        
        # Prepare metadata with UI state
        metadata = {
            "compliance_enabled": self.add_special_message,
            "selected_prompt_index": self.selected_prompt_index,
            "selected_reasoning_index": self.selected_reasoning_index,
            "selected_model_index": self.selected_model_index
        }
        
        if self.current_chat_id:
            # Update existing chat
            self.chat_persistence.save_chat(self.session, self.current_chat_id, 
                                           update_timestamp=update_timestamp, metadata=metadata)
        else:
            # Create new chat
            self.current_chat_id = self.chat_persistence.save_chat(self.session, metadata=metadata)
    
    def _refresh_chat_history(self) -> None:
        """Refresh the chat history list."""
        # Clear existing items
        while self.chat_list_layout.count() > 1:  # Keep the stretch
            item = self.chat_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Load and display chats
        chats = self.chat_persistence.list_chats()
        for chat in chats:
            # Parse timestamp to make it human-readable
            try:
                dt = datetime.fromisoformat(chat["updated_at"])
                timestamp = dt.strftime("%b %d, %H:%M")
            except:
                timestamp = "Unknown"
            
            chat_item = ChatHistoryItem(
                chat_id=chat["id"],
                title=chat["title"],
                timestamp=timestamp,
                message_count=chat["message_count"]
            )
            chat_item.clicked.connect(self._load_chat)
            chat_item.deleteRequested.connect(self._delete_chat)
            self.chat_list_layout.insertWidget(self.chat_list_layout.count() - 1, chat_item)
    
    def _load_chat(self, chat_id: str) -> None:
        """Load a saved chat."""
        # Save current chat if needed (without updating timestamp)
        if self.current_chat_id and self.current_chat_id != chat_id and len(self.session.messages) > 1:
            self._save_current_chat(update_timestamp=False)
        
        # Load the selected chat
        chat_data = self.chat_persistence.load_chat(chat_id)
        if not chat_data:
            return
        
        # Reconstruct the session
        self.current_chat_id = chat_id
        self.session = ChatSession(
            base_system_prompt=chat_data["base_system_prompt"],
            model_name=chat_data["model_name"]
        )
        
        # Restore messages
        self.session.messages = []
        for msg_data in chat_data["messages"]:
            msg = ChatMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                thinking=msg_data.get("thinking"),
                images=msg_data.get("images", [])
            )
            self.session.messages.append(msg)
        
        # Restore metadata if available
        metadata = chat_data.get("metadata", {})
        
        # Restore compliance state - always set it, defaulting to False if not in metadata
        self.add_special_message = metadata.get("compliance_enabled", False)
        self.compliance_checkbox.setChecked(self.add_special_message)
        
        # Update UI to match loaded chat's settings
        # Block signals to prevent triggering change handlers
        self.prompt_combo.blockSignals(True)
        self.model_combo.blockSignals(True)
        self.reasoning_combo.blockSignals(True)
        
        try:
            # Find and set the system prompt
            if "selected_prompt_index" in metadata:
                self.selected_prompt_index = metadata["selected_prompt_index"]
                self.prompt_combo.setCurrentIndex(self.selected_prompt_index)
            else:
                # Fallback to searching by prompt text
                for i, prompt in enumerate(SYSTEM_PROMPTS):
                    if prompt["prompt"] == chat_data["base_system_prompt"]:
                        self.selected_prompt_index = i
                        self.prompt_combo.setCurrentIndex(i)
                        break
            
            # Find and set the model
            if "selected_model_index" in metadata:
                self.selected_model_index = metadata["selected_model_index"]
                self.model_combo.setCurrentIndex(self.selected_model_index)
            else:
                # Fallback to searching by model name
                for i, model in enumerate(config.AVAILABLE_MODELS):
                    if model["name"] == chat_data["model_name"]:
                        self.selected_model_index = i
                        self.model_combo.setCurrentIndex(i)
                        break
            
            # Restore reasoning index if available
            if "selected_reasoning_index" in metadata:
                self.selected_reasoning_index = metadata["selected_reasoning_index"]
                self.reasoning_combo.setCurrentIndex(self.selected_reasoning_index)
            
            # Update UI based on loaded model
            self._update_ui_for_model(self.selected_model_index)
            
        finally:
            # Re-enable signals
            self.prompt_combo.blockSignals(False)
            self.model_combo.blockSignals(False)
            self.reasoning_combo.blockSignals(False)
        
        # Render the loaded chat
        self._render_history()
        self._refresh_chat_history()
    
    def _delete_chat(self, chat_id: str) -> None:
        """Delete a chat after confirmation."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Chat",
            "Are you sure you want to delete this chat?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.chat_persistence.delete_chat(chat_id)
            
            # If deleting current chat, start a new one
            if self.current_chat_id == chat_id:
                self.current_chat_id = None
                self.session.reset_messages()
                self._render_history()
            
            self._refresh_chat_history()

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
        if hasattr(self, '_export_btn'):
            self._position_export_button()

    def _apply_responsive_sidebar(self) -> None:
        # Keep behavior simple: do not auto-hide to avoid flicker; rely on user toggle
        try:
            sidebar = self.splitter.widget(0)
            sidebar_visible = sidebar.isVisible()
            if self._expand_sidebar_btn:
                self._expand_sidebar_btn.setVisible(not sidebar_visible)
            if hasattr(self, 'sidebar_collapse_btn'):
                self.sidebar_collapse_btn.setVisible(sidebar_visible)
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
    
    def _position_export_button(self) -> None:
        # Place in top-right corner of window
        if not hasattr(self, '_export_btn') or not self._export_btn:
            return
        geo = self.geometry()
        x = geo.width() - self._export_btn.width() - 12
        y = 12
        self._export_btn.move(x, y)

    # --- Image handling methods ---
    def _on_add_image(self) -> None:
        """Handle image selection."""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp)")
        
        if file_dialog.exec() == QtWidgets.QFileDialog.DialogCode.Accepted:
            file_paths = file_dialog.selectedFiles()
            for file_path in file_paths:
                self._add_image_from_path(file_path)
    
    def _add_image_from_path(self, file_path: str) -> None:
        """Add an image from file path."""
        try:
            # Read and encode the image
            with open(file_path, "rb") as f:
                image_data = f.read()
            
            # Determine content type
            ext = file_path.lower().split('.')[-1]
            content_type_map = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'gif': 'image/gif',
                'bmp': 'image/bmp',
                'webp': 'image/webp'
            }
            content_type = content_type_map.get(ext, 'image/jpeg')
            
            # Encode to base64
            import base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # Add to selected images
            self.selected_images.append({
                "data": base64_data,
                "type": content_type,
                "path": file_path
            })
            
            # Show preview
            self._update_image_preview()
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")
    
    def _update_image_preview(self) -> None:
        """Update the image preview container."""
        # Clear existing previews
        while self.image_preview_inner_layout.count() > 1:  # Keep the stretch
            item = self.image_preview_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add new previews
        for i, image_info in enumerate(self.selected_images):
            preview_widget = ImagePreviewWidget(i, image_info)
            # Use lambda with default argument to capture the current index
            preview_widget.removeRequested.connect(lambda idx, i=i: self._remove_image(i))
            preview_widget.imageClicked.connect(self._show_full_image)
            self.image_preview_inner_layout.insertWidget(i, preview_widget)
        
        # Show/hide preview container
        self.image_preview_container.setVisible(len(self.selected_images) > 0)
    

    
    def _remove_image(self, index: int) -> None:
        """Remove an image from the selection."""
        if 0 <= index < len(self.selected_images):
            self.selected_images.pop(index)
            self._update_image_preview()
    
    def _clear_selected_images(self) -> None:
        """Clear all selected images."""
        self.selected_images.clear()
        self._update_image_preview()
    
    def _show_full_image(self, pixmap: QtGui.QPixmap) -> None:
        """Show full size image in a dialog."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Image")
        dialog.setModal(True)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scrollable image viewer
        scroll = QtWidgets.QScrollArea()
        image_label = QtWidgets.QLabel()
        image_label.setPixmap(pixmap)
        scroll.setWidget(image_label)
        
        layout.addWidget(scroll)
        
        # Set dialog size to 80% of screen or image size, whichever is smaller
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        max_width = int(screen.width() * 0.8)
        max_height = int(screen.height() * 0.8)
        
        dialog_width = min(pixmap.width() + 20, max_width)
        dialog_height = min(pixmap.height() + 20, max_height)
        dialog.resize(dialog_width, dialog_height)
        
        dialog.exec()
    
    def _handle_paste(self) -> bool:
        """Handle paste event to check for images in clipboard."""
        # Check if current model supports images
        model_info = config.AVAILABLE_MODELS[self.selected_model_index]
        if not model_info.get("supports_images", False):
            return False  # Let default paste handle text
        
        clipboard = QtWidgets.QApplication.clipboard()
        mime_data = clipboard.mimeData()
        
        # Check if clipboard contains an image
        if mime_data.hasImage():
            image = clipboard.image()
            if not image.isNull():
                # Convert QImage to base64
                byte_array = QtCore.QByteArray()
                buffer = QtCore.QBuffer(byte_array)
                buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
                
                # Save as PNG by default
                image.save(buffer, "PNG")
                buffer.close()
                
                # Convert to base64
                import base64
                base64_data = base64.b64encode(byte_array.data()).decode('utf-8')
                
                # Add to selected images
                self.selected_images.append({
                    "data": base64_data,
                    "type": "image/png",
                    "path": "clipboard_image.png"
                })
                
                # Update preview
                self._update_image_preview()
                
                # Show a brief toast
                self._toast.show_message("Image pasted from clipboard")
                
                return True  # We handled the paste
        
        # Check if clipboard contains image URLs or file paths
        if mime_data.hasUrls():
            urls = mime_data.urls()
            images_added = 0
            
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    # Check if it's an image file
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        try:
                            self._add_image_from_path(file_path)
                            images_added += 1
                        except Exception:
                            pass  # Silently skip invalid files
            
            if images_added > 0:
                self._toast.show_message(f"{images_added} image{'s' if images_added > 1 else ''} pasted")
                return True
        
        return False  # Let default paste behavior handle text
    
    def _export_chat(self) -> None:
        """Export the current chat conversation."""
        if len(self.session.messages) <= 1:  # Only system message
            QtWidgets.QMessageBox.information(self, "Export", "No conversation to export.")
            return
        
        # Ask user for export format
        format_dialog = QtWidgets.QDialog(self)
        format_dialog.setWindowTitle("Export Format")
        format_dialog.setModal(True)
        
        layout = QtWidgets.QVBoxLayout(format_dialog)
        layout.setSpacing(12)
        
        label = QtWidgets.QLabel("Choose export format:")
        layout.addWidget(label)
        
        # Format radio buttons
        self.export_markdown_radio = QtWidgets.QRadioButton("Markdown (.md)")
        self.export_markdown_radio.setChecked(True)
        self.export_json_radio = QtWidgets.QRadioButton("JSON (.json)")
        self.export_txt_radio = QtWidgets.QRadioButton("Plain Text (.txt)")
        self.export_pdf_radio = QtWidgets.QRadioButton("PDF (.pdf)")
        
        layout.addWidget(self.export_markdown_radio)
        layout.addWidget(self.export_json_radio)
        layout.addWidget(self.export_txt_radio)
        layout.addWidget(self.export_pdf_radio)
        
        # Options
        self.export_include_system = QtWidgets.QCheckBox("Include system prompt")
        self.export_include_thinking = QtWidgets.QCheckBox("Include thinking/reasoning")
        self.export_include_thinking.setChecked(True)
        
        layout.addWidget(QtWidgets.QLabel())  # Spacer
        layout.addWidget(self.export_include_system)
        layout.addWidget(self.export_include_thinking)
        
        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(format_dialog.accept)
        buttons.rejected.connect(format_dialog.reject)
        layout.addWidget(buttons)
        
        if format_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        
        # Determine file extension and format
        if self.export_markdown_radio.isChecked():
            ext = "md"
            format_type = "markdown"
        elif self.export_json_radio.isChecked():
            ext = "json"
            format_type = "json"
        elif self.export_pdf_radio.isChecked():
            ext = "pdf"
            format_type = "pdf"
        else:
            ext = "txt"
            format_type = "text"
        
        # Get save file path
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setDefaultSuffix(ext)
        file_dialog.setNameFilter(f"{format_type.title()} files (*.{ext})")
        
        # Generate default filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title = self.chat_persistence._get_chat_title(self.session.messages).replace("...", "")[:30]
        # Clean title for filename
        import re
        clean_title = re.sub(r'[^\w\s-]', '', title).strip()
        clean_title = re.sub(r'[-\s]+', '-', clean_title)
        default_name = f"chat_{clean_title}_{timestamp}.{ext}"
        file_dialog.selectFile(default_name)
        
        if file_dialog.exec() != QtWidgets.QFileDialog.DialogCode.Accepted:
            return
        
        file_path = file_dialog.selectedFiles()[0]
        
        try:
            # Export based on format
            if format_type == "pdf":
                # PDF export uses a different approach
                self._export_as_pdf(
                    file_path,
                    include_system=self.export_include_system.isChecked(),
                    include_thinking=self.export_include_thinking.isChecked()
                )
            else:
                # Text-based exports
                if format_type == "markdown":
                    content = self._export_as_markdown(
                        include_system=self.export_include_system.isChecked(),
                        include_thinking=self.export_include_thinking.isChecked()
                    )
                elif format_type == "json":
                    content = self._export_as_json(
                        include_system=self.export_include_system.isChecked(),
                        include_thinking=self.export_include_thinking.isChecked()
                    )
                else:
                    content = self._export_as_text(
                        include_system=self.export_include_system.isChecked(),
                        include_thinking=self.export_include_thinking.isChecked()
                    )
                
                # Write to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            self._toast.show_message(f"Chat exported to {file_path.split('/')[-1]}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export chat: {str(e)}")
    
    def _export_as_markdown(self, include_system: bool, include_thinking: bool) -> str:
        """Export chat as markdown."""
        from core.m2m_formatter import is_m2m_format, parse_m2m_output, format_m2m_to_markdown
        
        lines = []
        
        # Add header
        lines.append(f"# Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n**Model:** {self.session.model_name}")
        lines.append(f"**System Prompt:** {SYSTEM_PROMPTS[self.selected_prompt_index]['name']}\n")
        
        # Check if M2M system prompt is being used
        is_m2m_prompt = (self.selected_prompt_index < len(SYSTEM_PROMPTS) and 
                        SYSTEM_PROMPTS[self.selected_prompt_index]["name"] == "M2M")
        
        # Add messages
        for msg in self.session.messages:
            if msg.role == "system" and not include_system:
                continue
            
            if msg.role == "user":
                lines.append(f"## User\n")
                lines.append(msg.content)
                if msg.images:
                    lines.append(f"\n*[{len(msg.images)} image(s) attached]*")
            elif msg.role == "assistant":
                lines.append(f"\n## Assistant\n")
                if include_thinking and msg.thinking:
                    lines.append("### Thinking\n")
                    lines.append(f"```\n{msg.thinking}\n```\n")
                
                # Apply M2M formatting if needed
                content = msg.content
                if is_m2m_prompt and is_m2m_format(content):
                    parsed_data = parse_m2m_output(content)
                    if parsed_data:
                        content = format_m2m_to_markdown(parsed_data)
                
                lines.append(content)
            elif msg.role == "system" and include_system:
                lines.append(f"## System\n")
                lines.append(f"```\n{msg.content}\n```")
            
            lines.append("")  # Empty line between messages
        
        return "\n".join(lines)
    
    def _export_as_json(self, include_system: bool, include_thinking: bool) -> str:
        """Export chat as JSON."""
        import json
        
        data = {
            "exported_at": datetime.now().isoformat(),
            "model": self.session.model_name,
            "system_prompt": SYSTEM_PROMPTS[self.selected_prompt_index]["name"],
            "messages": []
        }
        
        for msg in self.session.messages:
            if msg.role == "system" and not include_system:
                continue
            
            msg_data = {
                "role": msg.role,
                "content": msg.content
            }
            
            if include_thinking and msg.thinking:
                msg_data["thinking"] = msg.thinking
            
            if msg.images:
                # Only include image metadata, not the actual base64 data
                msg_data["images"] = [{"type": img["type"], "path": img.get("path", "unknown")} for img in msg.images]
            
            data["messages"].append(msg_data)
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _export_as_text(self, include_system: bool, include_thinking: bool) -> str:
        """Export chat as plain text."""
        from core.m2m_formatter import is_m2m_format, parse_m2m_output, format_m2m_to_markdown
        
        lines = []
        
        # Add header
        lines.append(f"Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Model: {self.session.model_name}")
        lines.append(f"System Prompt: {SYSTEM_PROMPTS[self.selected_prompt_index]['name']}")
        lines.append("=" * 60)
        lines.append("")
        
        # Check if M2M system prompt is being used
        is_m2m_prompt = (self.selected_prompt_index < len(SYSTEM_PROMPTS) and 
                        SYSTEM_PROMPTS[self.selected_prompt_index]["name"] == "M2M")
        
        # Add messages
        for msg in self.session.messages:
            if msg.role == "system" and not include_system:
                continue
            
            if msg.role == "user":
                lines.append("USER:")
                lines.append(msg.content)
                if msg.images:
                    lines.append(f"[{len(msg.images)} image(s) attached]")
            elif msg.role == "assistant":
                lines.append("\nASSISTANT:")
                if include_thinking and msg.thinking:
                    lines.append("[THINKING]")
                    lines.append(msg.thinking)
                    lines.append("[/THINKING]")
                
                # Apply M2M formatting if needed
                content = msg.content
                if is_m2m_prompt and is_m2m_format(content):
                    parsed_data = parse_m2m_output(content)
                    if parsed_data:
                        # For plain text, convert M2M to readable format
                        content = format_m2m_to_markdown(parsed_data)
                        # Strip markdown formatting for plain text
                        content = content.replace('**', '').replace('# ', '').replace('## ', '')
                
                lines.append(content)
            elif msg.role == "system" and include_system:
                lines.append("SYSTEM:")
                lines.append(msg.content)
            
            lines.append("-" * 40)
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_as_pdf(self, file_path: str, include_system: bool, include_thinking: bool) -> None:
        """Export chat as PDF using Qt's printing functionality."""
        from PySide6.QtPrintSupport import QPrinter
        from PySide6.QtGui import QTextDocument, QTextCursor
        from core.m2m_formatter import is_m2m_format, parse_m2m_output, format_m2m_to_markdown
        
        # Create printer and set output format
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(file_path)
        printer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.A4))
        printer.setPageMargins(QtCore.QMarginsF(20, 20, 20, 20), QtGui.QPageLayout.Unit.Millimeter)
        
        # Create document
        document = QTextDocument()
        cursor = QTextCursor(document)
        
        # Set document styles
        document.setDefaultStyleSheet("""
            body { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; font-size: 11pt; color: #333; }
            h1 { font-size: 20pt; font-weight: bold; margin-bottom: 12pt; }
            h2 { font-size: 16pt; font-weight: bold; margin-top: 16pt; margin-bottom: 8pt; color: #2e7d32; }
            h3 { font-size: 14pt; font-weight: bold; margin-top: 12pt; margin-bottom: 6pt; }
            p { margin-bottom: 8pt; line-height: 1.5; }
            .meta { color: #666; font-size: 10pt; }
            .user { color: #1976d2; }
            .assistant { color: #2e7d32; }
            .thinking { background-color: #f5f5f5; padding: 8pt; border-left: 3pt solid #ccc; font-style: italic; }
            pre { background-color: #f8f8f8; padding: 8pt; border: 1pt solid #ddd; font-family: monospace; }
            ul, li { margin-bottom: 4pt; }
            b { font-weight: bold; }
        """)
        
        # Add header
        cursor.insertHtml(f"<h1>Chat Export</h1>")
        cursor.insertHtml(f"<p class='meta'><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>")
        cursor.insertHtml(f"<b>Model:</b> {self.session.model_name}<br/>")
        cursor.insertHtml(f"<b>System Prompt:</b> {SYSTEM_PROMPTS[self.selected_prompt_index]['name']}</p>")
        cursor.insertHtml("<hr/>")
        
        # Check if M2M system prompt is being used
        is_m2m_prompt = (self.selected_prompt_index < len(SYSTEM_PROMPTS) and 
                        SYSTEM_PROMPTS[self.selected_prompt_index]["name"] == "M2M")
        
        # Add messages
        for msg in self.session.messages:
            if msg.role == "system" and not include_system:
                continue
            
            if msg.role == "user":
                cursor.insertHtml(f"<h2 class='user'>User</h2>")
                # Convert content to HTML, preserving line breaks
                content_html = html.escape(msg.content).replace('\n', '<br/>')
                cursor.insertHtml(f"<p>{content_html}</p>")
                if msg.images:
                    cursor.insertHtml(f"<p class='meta'><i>[{len(msg.images)} image(s) attached]</i></p>")
                    
            elif msg.role == "assistant":
                cursor.insertHtml(f"<h2 class='assistant'>Assistant</h2>")
                if include_thinking and msg.thinking:
                    cursor.insertHtml("<h3>Thinking</h3>")
                    thinking_html = html.escape(msg.thinking).replace('\n', '<br/>')
                    cursor.insertHtml(f"<div class='thinking'>{thinking_html}</div>")
                
                # Apply M2M formatting if needed
                content = msg.content
                if is_m2m_prompt and is_m2m_format(content):
                    parsed_data = parse_m2m_output(content)
                    if parsed_data:
                        content = format_m2m_to_markdown(parsed_data)
                
                # Convert markdown to basic HTML
                content_html = self._simple_markdown_to_html(content)
                cursor.insertHtml(f"<div>{content_html}</div>")
                
            elif msg.role == "system" and include_system:
                cursor.insertHtml("<h2>System</h2>")
                system_html = html.escape(msg.content).replace('\n', '<br/>')
                cursor.insertHtml(f"<pre>{system_html}</pre>")
            
            cursor.insertHtml("<br/>")
        
        # Print to PDF
        document.print_(printer)
    
    def _simple_markdown_to_html(self, markdown_text: str) -> str:
        """Convert basic markdown to HTML for PDF export."""
        import re
        
        # Escape HTML first
        html_text = html.escape(markdown_text)
        
        # Convert code blocks
        html_text = re.sub(r'```(.*?)```', r'<pre>\1</pre>', html_text, flags=re.DOTALL)
        
        # Convert inline code
        html_text = re.sub(r'`([^`]+)`', r'<code>\1</code>', html_text)
        
        # Convert headers
        html_text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_text, flags=re.MULTILINE)
        html_text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_text, flags=re.MULTILINE)
        html_text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_text, flags=re.MULTILINE)
        
        # Convert lists
        lines = html_text.split('\n')
        in_list = False
        processed_lines = []
        
        for line in lines:
            # Check if this is a list item
            if re.match(r'^\s*[-*]\s+', line):
                if not in_list:
                    processed_lines.append('<ul>')
                    in_list = True
                # Convert list item
                item_text = re.sub(r'^\s*[-*]\s+', '', line)
                processed_lines.append(f'<li>{item_text}</li>')
            else:
                # End list if we were in one
                if in_list and line.strip():
                    processed_lines.append('</ul>')
                    in_list = False
                processed_lines.append(line)
        
        # Close any open list
        if in_list:
            processed_lines.append('</ul>')
        
        html_text = '\n'.join(processed_lines)
        
        # Convert bold
        html_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', html_text)
        
        # Convert italic
        html_text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', html_text)
        
        # Convert line breaks
        html_text = html_text.replace('\n', '<br/>')
        
        return html_text


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


