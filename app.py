# Streamlit app for AI-labeling interface
# streamlit run app.py to run

import streamlit as st
import pandas as pd
from ai_assisted_coding_final import assistant
from openai import OpenAI
import time
import getpass
import os
from ai_assisted_coding_final.interactive_labeling import *
import hashlib

os.environ["OPENAI_API_KEY"] = "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"

# Initialize OpenAI assistant manager

if 'assistant_manager' not in st.session_state:
    client = OpenAI()  # Initialize your OpenAI client here
    st.session_state['assistant_manager'] = assistant.OpenAIAssistantManager(client)
    st.session_state['assistant_manager'].retrieve_assistant("asst_80dZeuENgjizYtopIa5Z2qtN")


# HARD CODED VARIABLES
initial_batch_size = 5

def update_context_with_labels(context_items):
    context_str = "Here are more examples of how to classify utterances:"
    for text, label in context_items:
        context_str += f"\nuser: '{text}'\nassistant: '{label}'"
    context_str += "\nI am going to provide several more sentences. Only answer with the following three latter labels: OTR, PRS, REP, NEU"
    return context_str

def label_data_st(unlabeled_text, default_label):
    """
    Streamlit version of label_data to get user input for labeling.

    Args:
    unlabeled_text (str): The text data that needs labeling.
    default_label (str): Default label to show in select box.

    Returns:
    str: The label selected by the user.
    """
    return st.selectbox(f"Label the following line: {unlabeled_text}", ["NEU", "OTR", "PRS", "REP"], index=["NEU", "OTR", "PRS", "REP"].index(default_label))

def get_user_labels_st(batch, assistant_manager, context):
    predictions = process_lines(batch, assistant_manager, context)
    labeled_data = []
    correct_responses = 0

    for text, prediction in predictions:
        key_suffix = hashlib.md5(text.encode()).hexdigest()  # Create a unique suffix for the key
        st.write(f"Predicted for '{text}': {prediction}")
        user_decision = st.radio(f"Do you agree with this label for '{text}'?", ('Yes', 'No'), key=f"decision_{key_suffix}")
        correct_label = label_data_st(text, prediction) if user_decision == 'No' else prediction
        correct_responses += (correct_label == prediction)
        st.session_state['formatted_context'] += f"\nuser: '{text}'\nassistant: '{correct_label}'"
        print("In Get User Labels: Formatted Context:", st.session_state['formatted_context'])
        labeled_data.append((text, correct_label))

    return labeled_data, correct_responses


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

        # Initialize session state variables if they don't exist
    if 'data' not in st.session_state:
        st.session_state['data'] = None
        st.session_state['context'] = []
        st.session_state['formatted_context'] = "" 
        st.session_state['index'] = 0
        st.session_state['initial_labels_submitted'] = False
        st.session_state['batch_size'] = initial_batch_size  # Define this value
        st.session_state['accuracy_scores'] = []
        st.session_state['all_labeled_data'] = []

    if uploaded_file is not None and st.session_state['data'] is None:
        st.session_state['data'] = load_data(uploaded_file)


    if st.session_state['data'] is not None:

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
                unique_key = f"label_button_{st.session_state['index']}_{option}"
                if cols[i].button(f"{option}", key=unique_key):
                    st.session_state['context'].append((text, option))
                    st.session_state['index'] += 1

            # for i, option in enumerate(label_options):
            #     unique_key = f"label_button_{st.session_state.index}_{option}"
            #     if cols[i].button(f"{option}", key=unique_key):
            #         st.session_state['context'].append((text, option))
            #         st.session_state['index'] += 1
            #         #labeled = True
            
            # for i, option in enumerate(label_options):
            #     unique_key = f"label_button_{st.session_state.index}_{option}"
            #     if cols[i].button(f"{option}", key=unique_key):
            #         st.session_state['labeled_data'].append((text, option))
            #         st.session_state['index'] += 1


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

        if st.button('Submit Labels'):        
            formatted_context = update_context_with_labels(st.session_state['context'])
            st.session_state['formatted_context'] = formatted_context
            st.session_state['context'] = [] 
            #context = update_context_with_labels()
            # You can now use this context for further processing or model training
            st.success('Labels submitted successfully!')
            st.session_state['initial_labels_submitted'] = True
        
        if st.session_state['initial_labels_submitted']:
        # Assuming 'index' and 'batch_size' are initialized in session state
            while st.session_state['index'] < len(st.session_state['data']):
                print("Before batch", "Data",  st.session_state['data'],type(st.session_state['data']) ,"Index", st.session_state['index'])
                batch, actual_batch_size = process_batch(st.session_state['data']['Text'].tolist(), st.session_state['index'], st.session_state['batch_size'], []) # not keeping track of batch sizes yet 
                print("Type:", type(batch))
                print("Batch:", batch)
                print("Batch Size:", actual_batch_size)
                print("First Formatted Context:", st.session_state['formatted_context'])
                labeled_data, correct_responses = get_user_labels_st(batch, st.session_state['assistant_manager'], st.session_state['formatted_context'])

                if st.button("Submit Batch"):
                    st.session_state['all_labeled_data'].extend(labeled_data)
                    accuracy = calculate_accuracy(correct_responses, actual_batch_size)
                    st.session_state['accuracy_scores'].append(accuracy)
                    st.write(f"Current Batch Accuracy: {accuracy}")
                    st.session_state['batch_size'] = increase_batch_size(st.session_state['batch_size'], accuracy)
                    st.session_state['index'] += actual_batch_size
                    break
        


        if st.session_state['index'] >= len(st.session_state['data']):
            st.markdown("<div class='big-font'>All sentences labeled!</div>", unsafe_allow_html=True)

        # progress_bar = st.progress(0)
        # for i in range(100):
        #     progress_bar.progress(i + 1)

        # if st.session_state['labeled_data']:
        #     st.line_chart(st.session_state['accuracy_scores'])

        if st.button("Save Data"):
            st.session_state['data'].to_csv("labeled_data.csv")
            st.success("Data saved successfully!")

        if st.button("Export Data"):
            st.download_button('Download CSV', data.to_csv().encode('utf-8'), 'labeled_data.csv', 'text/csv')

if __name__ == "__main__":
    main()



            # while st.session_state['index'] < len(data):
            #     batch, actual_batch_size = process_batch(data, st.session_state['index'], st.session_state['batch_size'], [])
            #     labeled_data, correct_responses = get_user_labels_st(batch, assistant_manager, context)
            #     if st.button("Submit Batch"):

            #         # Extend the all labeled data with new labeled data
            #         st.session_state['all_labeled_data'].extend(labeled_data)

            #         # Calculate and update accuracy
            #         accuracy = calculate_accuracy(correct_responses, actual_batch_size)
            #         st.session_state['accuracy_scores'].append(accuracy)
            #         st.write(f"Current Batch Accuracy: {accuracy}")

            #         # Update batch size based on accuracy
            #         st.session_state['batch_size'] = increase_batch_size(st.session_state['batch_size'], accuracy)

            #         # Update index for the next batch
            #         st.session_state['index'] += actual_batch_size

            #         # Show updated information or progress
            #         st.write(f"Updated Batch Size for Next Batch: {st.session_state['batch_size']}")
            #         st.write("Proceeding to next batch...")

            #         break 




    # if uploaded_file is not None and st.session_state['data'] is None:
    #     data = load_data(uploaded_file)
    #     st.session_state.setdefault('data', data)
    #     st.session_state.setdefault('context', [])
    #     st.session_state.setdefault('index', 0)
    #     st.session_state['initial_labels_submitted'] = False
    #     st.session_state['batch_size'] = initial_batch_size 

    # if uploaded_file is not None:
    #     data = load_data(uploaded_file)
    #     st.session_state.setdefault('data', data)
    #     st.session_state.setdefault('labeled_data', [])
    #     st.session_state.setdefault('index', 0)
    #     st.session_state.setdefault('accuracy_scores', [])

