"""Core chat service decoupled from any UI framework.

This module prepares model messages and options, manages conversation state,
and provides small helpers that a UI layer (Streamlit, Qt, etc.) can use to
perform streaming with the Ollama client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any

import config


REASONING_OPTIONS = ["High", "Medium", "Low", "None"]


@dataclass
class ChatMessage:
    role: str
    content: str
    thinking: str | None = None


@dataclass
class ChatSession:
    base_system_prompt: str
    model_name: str = config.MODEL_NAME
    messages: List[ChatMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            self.reset_messages()

    def reset_messages(self) -> None:
        self.messages = [ChatMessage(role="system", content=self.base_system_prompt)]

    def build_stream_payload(
        self,
        user_input: str,
        add_special_message: bool,
        reasoning_effort: str,
    ) -> tuple[list[dict[str, str]], dict[str, Any], str]:
        """Return (model_messages, model_options, message_to_send)."""
        if add_special_message:
            message_to_send = config.COMPLIANCE_PROMPT.format(user_input=user_input)
        else:
            message_to_send = user_input

        # If reasoning is "None", append the no-think prompt to the message being sent
        # Skip this for non-reasoning models (empty string reasoning_effort)
        if reasoning_effort == "None":
            message_to_send += config.NOTHINK_PROMPT

        model_options: dict[str, Any] = {}
        if reasoning_effort in ["High", "Medium", "Low"]:
            model_options["reasoning_effort"] = reasoning_effort.lower()

        # Base history: if the last message is a user message (often just appended
        # by the UI), exclude it so we can send the possibly transformed
        # message_to_send instead (e.g., compliance or no-think).
        if self.messages and self.messages[-1].role == "user":
            base_history = self.messages[:-1]
        else:
            base_history = self.messages

        # Convert internal messages to API format and append current user message
        model_messages = [
            {"role": m.role, "content": m.content}
            for m in base_history
            if m.role != "assistant" or (m.content or m.thinking)
        ] + [{"role": "user", "content": message_to_send}]

        return model_messages, model_options, message_to_send

    def add_user_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str, thinking: str | None) -> None:
        self.messages.append(
            ChatMessage(role="assistant", content=content, thinking=thinking or None)
        )


