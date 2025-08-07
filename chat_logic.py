# chat_logic.py
import streamlit as st
from ollama import chat
import config  # Imports our config file
from core.chat_service import ChatSession, ChatMessage

def process_and_stream_response(user_input, add_special_message, reasoning_effort):
    """
    Processes user input, sends it to the Ollama model, and streams the response.
    """
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=config.USER_AVATAR):
        st.markdown(user_input)

    # Build payload via UI-agnostic core service so both Streamlit and Desktop share logic
    base_system_prompt = st.session_state.messages[0]["content"] if st.session_state.messages else ""
    session = ChatSession(base_system_prompt=base_system_prompt, model_name=config.MODEL_NAME)
    # Mirror Streamlit history (skip the system message which session already has)
    for m in st.session_state.messages[1:]:
        session.messages.append(
            ChatMessage(role=m["role"], content=m.get("content", ""), thinking=m.get("thinking"))
        )

    model_messages, model_options, message_to_send = session.build_stream_payload(
        user_input=user_input,
        add_special_message=add_special_message,
        reasoning_effort=reasoning_effort,
    )

    with st.chat_message("assistant", avatar=config.ASSISTANT_AVATAR_PATH):
        thinking_container = st.container()
        response_placeholder = st.empty()
        full_thinking_content = ""
        full_response_content = ""
        has_thinking_content = False

        stream = chat(
            model=config.MODEL_NAME,
            messages=model_messages,
            stream=True,
            options=model_options # Correct way to pass model-specific parameters
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
                status.update(state="complete", expanded=False)
                thinking_container.empty()
        
        response_placeholder.markdown(full_response_content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response_content,
        "thinking": full_thinking_content
    })