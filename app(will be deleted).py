import streamlit as st
import pandas as pd
from ai_assisted_coding_final.AskTheDataset import *

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'reset_query' not in st.session_state:
    st.session_state.reset_query = False

# Function to handle CSV files
def process_csv(uploaded_files):
    try:
        if uploaded_files is not None:
            df = pd.concat([pd.read_csv(file) for file in uploaded_files])
            return df
        return None
    except:
        st.write("Please upload files")

# Callback function for the submit button
def process_query():
    if api_key and df is not None and user_query:
        csv_combined = df.to_string()
        response = ask_gpt(user_query, csv_combined, api_key)
        st.session_state.history.append({"question": user_query, "response": response})
        st.session_state.reset_query = True  # Flag to reset the query input

# Streamlit UI components
st.title("CSV and Chatbot Interface")

# API Key Input
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# File Uploader
uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True)
df = process_csv(uploaded_files)

# User Query Input
if not st.session_state.reset_query:
    user_query = st.text_input("Ask a question about the dataset", key="query")
else:
    user_query = ""
    st.session_state.reset_query = False  # Reset the flag

# Submit Button
if st.button("Submit", on_click=process_query):
    pass  # The processing is handled in the process_query function

# Display Chat History
st.header("Chat History")
for chat in reversed(st.session_state.history):
    st.text_area(label="", value=f"Q: {chat['question']}\nA: {chat['response']}", height=100, disabled=True)
