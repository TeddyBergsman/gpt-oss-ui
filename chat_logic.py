# chat_logic.py
import streamlit as st
from ollama import chat
import config # Imports our config file

def process_and_stream_response(user_input, add_special_message):
    """
    Processes user input, sends it to the Ollama model, and streams the response.
    """
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=config.USER_AVATAR):
        st.markdown(user_input)

    # This part remains identical. It uses the COMPLIANCE_PROMPT variable,
    # oblivious to its origin (file or string), making our code robust.
    if add_special_message:
        message_to_send = config.COMPLIANCE_PROMPT.format(user_input=user_input)
    else:
        message_to_send = user_input

    model_messages = st.session_state.messages[:-1] + [{"role": "user", "content": message_to_send}]

    with st.chat_message("assistant", avatar=config.ASSISTANT_AVATAR_PATH):
        thinking_container = st.container()
        response_placeholder = st.empty()
        full_thinking_content = ""
        full_response_content = ""
        has_thinking_content = False

        stream = chat(
            model=config.MODEL_NAME,
            messages=model_messages,
            stream=True
        )
        
        with thinking_container, st.status("Reasoning...", expanded=True) as status:
            thinking_placeholder = st.empty()
            for chunk in stream:
                if thinking_content := chunk["message"].get("thinking"):
                    has_thinking_content = True
                    full_thinking_content += thinking_content
                    thinking_placeholder.markdown(full_thinking_content + " ▌")

                if response_content := chunk["message"].get("content"):
                    full_response_content += response_content
                    response_placeholder.markdown(full_response_content + " ▌")

            if has_thinking_content:
                status.update(label="Reasoning", state="complete", expanded=False)
                thinking_placeholder.markdown(full_thinking_content)
            else:
                status.update(state="complete")
        
        if not has_thinking_content:
            thinking_container.empty()
            
        response_placeholder.markdown(full_response_content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response_content,
        "thinking": full_thinking_content
    })