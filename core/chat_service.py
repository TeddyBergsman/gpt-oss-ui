"""Core chat service decoupled from any UI framework.

This module prepares model messages and options, manages conversation state,
and provides small helpers that a UI layer (Streamlit, Qt, etc.) can use to
perform streaming with the Ollama client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from copy import deepcopy

import config


REASONING_OPTIONS = ["High", "Medium", "Low", "None"]


@dataclass
class ChatMessage:
    role: str
    content: str
    thinking: str | None = None
    images: List[Dict[str, str]] = field(default_factory=list)  # List of {"data": base64_str, "type": mime_type}


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
        model_messages = []
        for m in base_history:
            if m.role != "assistant" or (m.content or m.thinking):
                msg = {"role": m.role, "content": m.content}
                # Add images if present - Ollama expects base64 strings directly
                if m.images:
                    msg["images"] = [img["data"] for img in m.images]
                model_messages.append(msg)
        
        # Add current user message
        user_msg = {"role": "user", "content": message_to_send}
        # Get images from the last user message if any
        print(f"[chat_service] Debug - Number of messages: {len(self.messages)}")
        if self.messages:
            last_msg = self.messages[-1]
            print(f"[chat_service] Debug - Last message: role={last_msg.role}, has content={bool(last_msg.content)}, images count={len(last_msg.images)}")
            if last_msg.images:
                print(f"[chat_service] Debug - First image data length: {len(last_msg.images[0]['data']) if last_msg.images[0].get('data') else 'No data key'}")
        
        if self.messages and self.messages[-1].role == "user" and self.messages[-1].images:
            user_msg["images"] = [img["data"] for img in self.messages[-1].images]
            print(f"[chat_service] Adding {len(self.messages[-1].images)} images to user message")
        else:
            print(f"[chat_service] No images to add. Last message role: {self.messages[-1].role if self.messages else 'No messages'}, has images: {bool(self.messages[-1].images) if self.messages and self.messages[-1].role == 'user' else False}")
        model_messages.append(user_msg)

        return model_messages, model_options, message_to_send

    def add_user_message(self, content: str, images: List[Dict[str, str]] | None = None) -> None:
        # Deep copy images to avoid reference issues when the original list is cleared
        images_copy = deepcopy(images) if images else []
        self.messages.append(ChatMessage(role="user", content=content, images=images_copy))

    def add_assistant_message(self, content: str, thinking: str | None) -> None:
        self.messages.append(
            ChatMessage(role="assistant", content=content, thinking=thinking or None)
        )


