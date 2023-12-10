import streamlit as st
import pandas as pd
import time
from ai_assisted_coding_final.AskTheDataset import *

st.title("CSV and Chatbot Interface")

MAX_HISTORY_LENGTH = 100  # Maximum number of messages in chat history

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Function to handle CSV files
def process_csv(uploaded_files):
    try:
        if uploaded_files is not None:
            df = pd.concat([pd.read_csv(file) for file in uploaded_files])
            return df
        return None
    except:
        st.error("Please upload valid CSV files")

# Function to process user query
def process_query(user_query):
    if st.session_state.api_key and st.session_state.df is not None and user_query:
        csv_combined = st.session_state.df.to_string()
        response = ask_gpt(user_query, csv_combined, st.session_state.api_key)
        
        # Simulate typing for assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Manage history length
        if len(st.session_state.history) >= MAX_HISTORY_LENGTH:
            st.session_state.history.pop(0)  # Remove the oldest message
        st.session_state.history.append({"role": "bot", "content": response})

# Display chat messages from history
def display_history():
    start_index = max(0, len(st.session_state.history) - MAX_HISTORY_LENGTH)
    for message in st.session_state.history[start_index:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

display_history()

# API Key Input
if not st.session_state.api_key:
    st.session_state.api_key = st.text_input("Enter your OpenAI API Key", type="password")

# File Uploader
if st.session_state.api_key and (st.session_state.df is None or st.session_state.df.empty):
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True)
    st.session_state.df = process_csv(uploaded_files)

# Chat Interface
if st.session_state.api_key and st.session_state.df is not None:
    user_query = st.chat_input("Ask a question about the dataset")
    if user_query:
        # Append user's question to history and manage history length
        if len(st.session_state.history) >= MAX_HISTORY_LENGTH:
            st.session_state.history.pop(0)
        st.session_state.history.append({"role": "user", "content": user_query})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process and respond to query
        process_query(user_query)
