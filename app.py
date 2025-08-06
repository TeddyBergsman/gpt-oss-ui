# main_app.py
import streamlit as st
import config
import ui_components
import chat_logic
import system_prompts

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Ollama Streaming Chat", layout="centered")

    ui_components.display_header()

    with st.sidebar:
        st.header("Settings")

        # --- Compliance Protocol ---
        add_special_message = st.checkbox("Compliance Protocol", key="special_message_toggle")

        # --- Persona Selection ---
        prompt_names = [p["name"] for p in system_prompts.SYSTEM_PROMPTS]
        if "prompt_index" not in st.session_state:
            st.session_state.prompt_index = 0
        
        selected_prompt_index = st.selectbox(
            "System Prompt:",
            range(len(prompt_names)),
            format_func=lambda i: prompt_names[i],
            key="prompt_selector"
        )
        base_prompt = system_prompts.SYSTEM_PROMPTS[selected_prompt_index]["prompt"]

        # --- Reasoning Effort Selection ---
        reasoning_options = ["High", "Medium", "Low", "None"]
        if "reasoning_index" not in st.session_state:
            st.session_state.reasoning_index = 0  # Default to "High"

        selected_reasoning_index = st.selectbox(
            "Reasoning Effort:",
            range(len(reasoning_options)),
            format_func=lambda i: reasoning_options[i],
            key="reasoning_selector"
        )
        reasoning_effort = reasoning_options[selected_reasoning_index]
    

    # --- System Prompt Construction ---
    # The reasoning effort is now passed as an API parameter in chat_logic.py,
    # so we no longer need to modify the system prompt text here.
    final_system_prompt = base_prompt

    # --- Chat Reset Logic ---
    if (st.session_state.prompt_index != selected_prompt_index or
        st.session_state.reasoning_index != selected_reasoning_index or
        "messages" not in st.session_state):
        
        st.session_state.messages = [
            {"role": "system", "content": final_system_prompt}
        ]
        st.session_state.prompt_index = selected_prompt_index
        st.session_state.reasoning_index = selected_reasoning_index

    # --- UI Display ---
    ui_components.display_chat_history()
    
    if user_input := st.chat_input("Type your message here..."):
        # Pass the reasoning_effort to the logic layer for processing
        chat_logic.process_and_stream_response(user_input, add_special_message, reasoning_effort)

if __name__ == "__main__":
    main()