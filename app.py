import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Basic page config
st.set_page_config(
    page_title="ChatGPT2 Demo",
    page_icon="🤖",
)

st.title("💬 ChatGPT2 Demo")

# Add a loading message while the model loads
with st.spinner('Loading model... This might take a minute...'):
    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        return model, tokenizer

    model, tokenizer = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Hugging Face") 