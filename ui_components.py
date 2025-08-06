# ui_components.py
import streamlit as st
import base64
import config

def display_header():
    """Displays the main header and logo of the application."""
    try:
        logo = base64.b64encode(open(config.ASSISTANT_AVATAR_PATH, "rb").read()).decode()
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h1 style='margin-bottom: 0.5rem;'>
                <img src="data:image/png;base64,{logo}" width="40" style="vertical-align: middle;">
            </h1>
        </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("<div style='text-align: center; margin-bottom: 2rem;'><h1>Chat</h1></div>", unsafe_allow_html=True)

def display_message(message):
    """Displays a single message from the chat history."""
    role = "user" if message["role"] == "user" else "assistant"
    avatar = config.USER_AVATAR if role == "user" else config.ASSISTANT_AVATAR_PATH
    
    with st.chat_message(role, avatar=avatar):
        if role == "assistant" and "thinking" in message and message["thinking"]:
            with st.expander("Reasoning", expanded=False):
                st.markdown(message["thinking"])
        st.markdown(message["content"])

def display_chat_history():
    """Displays all previous messages in the chat history."""
    for message in st.session_state.get("messages", []):
        if message["role"] != "system":
            display_message(message)