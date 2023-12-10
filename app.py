# Streamlit app for AI-labeling interface
# streamlit run app.py to run

import streamlit as st
import pandas as pd
from ai_assisted_coding_final import assistant
from openai import OpenAI
import time
import getpass
import os


os.environ["OPENAI_API_KEY"] = "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"

# Set OpenAI API Key
# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# Initialize OpenAI assistant manager
client = OpenAI()
assistant_manager = assistant.OpenAIAssistantManager(client)


def get_predictions_with_context(context, new_data, assistant_manager):
    predictions = []
    for data in new_data:
        prompt = context + f"\n\n{data}"
        thread, completed_run = assistant_manager.create_thread_and_run(prompt)
        response_page = assistant_manager.get_response()
        messages = [msg for msg in response_page]
        if messages:
            prediction = messages[-1]['content']['text']['value']
            predictions.append(prediction)
    return predictions

# def display_sentences_for_labeling(sentences, labels=None):
#     user_labels = []
#     for i, sentence in enumerate(sentences):
#         label = st.selectbox(f"Label the sentence: {sentence}", ['OTR', 'PRS', 'REP', 'NEU'], key=f"label_{i}")
#         user_labels.append(label)
#     return user_labels


@st.cache_resource
def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    st.title("Interactive AI Assisted Labeling")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Styling
    st.markdown("""
        <style>
            .big-font {
                font-size:20px !important;
                font-weight: bold;
            }
            .label-box {
                border: 2px solid #4CAF50;
                padding: 10px;
                margin-top: 10px;
                margin-bottom: 20px;
                background-color: #f1f8e9;
            }
            .labeled-sentence {
                font-size: 16px;
                color: #333;
                margin-right: 10px;
            }
            .edit-button {
                font-size: 12px;
                margin-left: 10px;
                color: #4CAF50;
                border: 1px solid #4CAF50;
                border-radius: 5px;
                padding: 2px 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.session_state.setdefault('data', data)
        st.session_state.setdefault('context', [])
        st.session_state.setdefault('index', 0)

        if st.session_state['index'] < len(st.session_state['data']):
            current_data = st.session_state['data'].iloc[st.session_state['index']]
            text = current_data['Text']
            st.markdown("<h3 style='font-weight: bold;'>Sentence to Label:</h3>", unsafe_allow_html=True)


            # label for box containing the sentence
            #st.markdown(f"<div class='big-font' style='border: 4px solid #ccc; padding: 10px;'>{text}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='border: 2px solid #ccc; background-color: #F5F5DC; padding: 10px; text-align: center; font-size: 18px;'>{text}</div><br>", unsafe_allow_html=True)


            # Horizontal buttons for labeling
            label_options = ["OTR", "PRS", "REP", "NEU"]
            cols = st.columns(len(label_options))
            # labeled = False

            for i, option in enumerate(label_options):
                unique_key = f"label_button_{st.session_state.index}_{option}"
                if cols[i].button(f"{option}", key=unique_key):
                    st.session_state['context'].append((text, option))
                    st.session_state['index'] += 1
                    #labeled = True


        # Display and edit labeled sentences
        st.markdown("<div class='label-box'>", unsafe_allow_html=True)
        for i, context_item in enumerate(st.session_state['context']):
            if isinstance(context_item, tuple) and len(context_item) == 2:
                sentence, label = context_item
                st.markdown(f"<div><span class='labeled-sentence'>{sentence} [{label}]</span>", unsafe_allow_html=True)
                if st.button("Edit", key=f"edit_{i}", args={'class': 'edit-button'}):
                    new_label = st.selectbox("Choose a new label", label_options, key=f"new_label_{i}")
                    st.session_state['context'][i] = (sentence, new_label)

        # important to keep this here, please do not change indentation!!
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state['index'] >= len(st.session_state['data']):
            st.markdown("<div class='big-font'>All sentences labeled!</div>", unsafe_allow_html=True)

        # progress_bar = st.progress(0)
        # for i in range(100):
        #     progress_bar.progress(i + 1)

        if st.button("Save Data"):
            st.session_state['data'].to_csv("labeled_data.csv")
            st.success("Data saved successfully!")

        if st.button("Export Data"):
            st.download_button('Download CSV', data.to_csv().encode('utf-8'), 'labeled_data.csv', 'text/csv')

if __name__ == "__main__":
    main()









