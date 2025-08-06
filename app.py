# main_app.py
import streamlit as st
import config
import ui_components
import chat_logic

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Ollama Streaming Chat", layout="centered")

    ui_components.display_header()

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        add_special_message = st.checkbox("Engage Compliance Protocol", key="special_message_toggle")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    ui_components.display_chat_history()
    
    if user_input := st.chat_input("Type your message here..."):
        chat_logic.process_and_stream_response(user_input, add_special_message)

if __name__ == "__main__":
    main()