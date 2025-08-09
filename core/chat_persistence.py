"""Chat persistence module for saving and loading chat sessions."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from core.chat_service import ChatMessage, ChatSession


class ChatPersistence:
    """Handles saving and loading chat sessions to/from disk."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize chat persistence with a storage directory."""
        if storage_dir is None:
            # Default to user's home directory
            self.storage_dir = Path.home() / ".gpt-oss-chats"
        else:
            self.storage_dir = Path(storage_dir)
        
        # Create directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_chat_id(self) -> str:
        """Generate a unique chat ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def _get_chat_title(self, messages: List[ChatMessage]) -> str:
        """Generate a title for the chat based on first user message."""
        for msg in messages:
            if msg.role == "user" and msg.content.strip():
                # Take first 50 characters of first user message
                title = msg.content.strip()[:50]
                if len(msg.content.strip()) > 50:
                    title += "..."
                return title
        return "New Chat"
    
    def save_chat(self, session: ChatSession, chat_id: Optional[str] = None, 
                  update_timestamp: bool = True, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a chat session to disk. Returns the chat ID."""
        if chat_id is None:
            chat_id = self._generate_chat_id()
            created_at = datetime.now().isoformat()
        else:
            # Load existing chat to preserve created_at and updated_at if not updating
            existing = self.load_chat(chat_id)
            created_at = existing["created_at"] if existing else datetime.now().isoformat()
        
        # Prepare chat data
        chat_data = {
            "id": chat_id,
            "title": self._get_chat_title(session.messages),
            "created_at": created_at,
            "updated_at": datetime.now().isoformat() if update_timestamp else (existing.get("updated_at", created_at) if existing else created_at),
            "model_name": session.model_name,
            "base_system_prompt": session.base_system_prompt,
            "messages": [asdict(msg) for msg in session.messages]
        }
        
        # Add any additional metadata
        if metadata:
            chat_data["metadata"] = metadata
        
        # Save to file
        file_path = self.storage_dir / f"{chat_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        return chat_id
    
    def load_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Load a chat session from disk."""
        file_path = self.storage_dir / f"{chat_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chat {chat_id}: {e}")
            return None
    
    def list_chats(self) -> List[Dict[str, Any]]:
        """List all saved chats with metadata."""
        chats = []
        
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                    # Include only metadata for listing
                    chats.append({
                        "id": chat_data["id"],
                        "title": chat_data["title"],
                        "created_at": chat_data["created_at"],
                        "updated_at": chat_data.get("updated_at", chat_data["created_at"]),
                        "message_count": len(chat_data.get("messages", [])) - 1  # Exclude system message
                    })
            except Exception as e:
                print(f"Error reading chat file {file_path}: {e}")
                continue
        
        # Sort by updated_at (most recent first)
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat from disk."""
        file_path = self.storage_dir / f"{chat_id}.json"
        
        try:
            if file_path.exists():
                file_path.unlink()
                return True
        except Exception as e:
            print(f"Error deleting chat {chat_id}: {e}")
        
        return False
    
    def update_chat_timestamp(self, chat_id: str) -> None:
        """Update the 'updated_at' timestamp of a chat."""
        chat_data = self.load_chat(chat_id)
        if chat_data:
            chat_data["updated_at"] = datetime.now().isoformat()
            file_path = self.storage_dir / f"{chat_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)
