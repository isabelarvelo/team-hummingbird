# Streamlit app for ai-labeling interface
# $ streamlit run app.py to run

import streamlit as st
import pandas as pd
from ai_assisted_coding_final import assistant
from openai import OpenAI

import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# Initialize OpenAI assistant manager
client = OpenAI()
assistant_manager = assistant.OpenAIAssistantManager(client)


def get_predictions_with_context(context, new_data, assistant_manager):
    # Function to get predictions for new data based on the context
    predictions = []
    for data in new_data:
        prompt = context + f"\n\n{data}"
        thread, completed_run = assistant_manager.create_thread_and_run(prompt)
        response_page = assistant_manager.get_response()
        messages = [msg for msg in response_page]
        if messages:
            prediction = messages[-1].content[0].text.value.split()[-1]
            predictions.append(prediction)
    return predictions

# Read data from CSV
@st.cache_resource
def load_data(file_path):
    return pd.read_csv(file_path)

# Main app
def main():
    st.title("Interactive AI Assisted Labeling")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if 'data' not in st.session_state:
            st.session_state['data'] = data
            st.session_state['context'] = ""
            st.session_state['index'] = 0

        # Display and label data
        if 'index' in st.session_state and st.session_state['index'] < len(st.session_state['data']):
            current_data = st.session_state['data'].iloc[st.session_state['index']]
            text = current_data['Text']
            st.write("Text to Label:", text)

            # User inputs label
            label = st.text_input("Label")
            if st.button("Submit Label"):
                # Update context
                st.session_state['context'] += f"\n{text}: {label}"
                st.session_state['index'] += 1

        # Show context for verification
        st.text_area("Current context (for verification)", st.session_state['context'], height=300)

        # Save context to CSV
        if st.button("Save Labeled Data"):
            with open("labeled_data.csv", "w") as f:
                f.write(st.session_state['context'])
            st.success("Data saved successfully!")

if __name__ == "__main__":
    main()
