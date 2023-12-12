import streamlit as st
from ai_assisted_coding_final.AskTheDataset import *
import time

st.title("Ask about the Dataset Page")

MAX_HISTORY_LENGTH = 100  # Maximum number of messages in chat history

# Welcome Message and Instructions
st.markdown("""
Welcome to the Data Analysis Chatbot! 

This interactive tool allows you to upload your CSV data files and ask questions about your data using natural language. 
Our AI-powered assistant will help you understand and analyze your data.

**Follow these simple steps to get started:**
1. **Enter your OpenAI API Key**: This is required to enable the AI-powered analysis. You can sign up for an account and register for an API key at https://openai.com/blog/openai-api
2. **Upload Your CSV Data**: You can upload one or more CSV files containing your data.
3. **Ask Questions**: Once your data is uploaded, simply type in your questions about the data in the chat interface. The AI assistant will respond with insights and analyses.

At any point during the chat, if you want to upload new data, you can upload new CSV files/ remove existing ones and the AI assistant will use the new data for analysis.

You can also clear the chat history at any point by clicking the "Clear Chat History" button.

Let's begin!
""")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = CSVFileManager()
if 'ai_handler' not in st.session_state:
    st.session_state.ai_handler = None
if 'df' not in st.session_state:  # Initialize 'df' in session state
    st.session_state.df = None

# Function to clear chat history
def clear_history():
    st.session_state.history = []

    
# API Key Input
if not st.session_state.api_key:
    api_key_input = st.text_input("Enter your OpenAI API Key", type="password")
    if st.button('Submit'):
        st.session_state.api_key = api_key_input
        if 'ai_handler' not in st.session_state or st.session_state.ai_handler is None:
            st.session_state.ai_handler = GPTQuestionsHandler(api_key_input)
        st.write("Please proceed to upload your CSV files")

# File Uploader
uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True)
if uploaded_files:
    st.session_state.data_manager.load_data(uploaded_files)
    st.session_state.df = st.session_state.data_manager.data_frame


# Function to process user query
def process_query(user_query):
    if st.session_state.ai_handler and st.session_state.df is not None and user_query:
        csv_combined = st.session_state.df.to_string()
        response = st.session_state.ai_handler.ask_gpt(user_query, csv_combined)
        
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

if st.button('Clear Chat History'):
    clear_history()
