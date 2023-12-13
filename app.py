# Streamlit app for AI-labeling interface
# streamlit run app.py to run

#IMPORT STATEMENTS 
import streamlit as st
import pandas as pd
from ai_assisted_coding_final import assistant
from openai import OpenAI
import time
import getpass
import os
from ai_assisted_coding_final.interactive_labeling import *
import hashlib
import altair as alt

os.environ["OPENAI_API_KEY"] = "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"

# HARD CODED VARIABLES
INITIAL_BATCH_SIZE = 5

def update_context_with_labels(context_items):
    print("inside of update_context_with_labels")
    context_str = "Example(s) of how to classify utterances:"
    for text, label in context_items:
        context_str += f"\nuser: '{text}'\nassistant: '{label}'"
    context_str += "\nOnly answer with one the following three-letter labels: OTR, PRS, REP, NEU"
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


def process_user_inputs_and_prepare_next_batch(batch_texts, predictions, context):
    print("Inside of process_user_inputs_and_prepare_next_batch")
    labeled_data = []
    correct_responses = 0

    #print("batch_texts", batch_texts)
    #print("predictions", predictions)

    for index, (text, original_prediction) in enumerate(zip(batch_texts, predictions)):
        key_suffix = f"{index}_{hashlib.md5(text.encode()).hexdigest()}"
        user_decision = st.session_state.get(f"decision_{key_suffix}", 'Yes')
        #print("user decision", user_decision)
        selected_label = st.session_state.get(f"correct_label_{key_suffix}", original_prediction[1])
        #print("selected label", selected_label)

        correct_label = selected_label if user_decision == 'No' else original_prediction[1]
        #print("correct label", correct_label)
        #print("append tuple", (text, correct_label))
        labeled_data.append((text, correct_label))
        

    # Update context
    for text, label in labeled_data:
        context += f"\nuser: '{text}'\nassistant: '{label}'"
    
    #print("inside of process user inputs and prepare next batch", "new labeled data", labeled_data, "context:", context)

    return labeled_data, context

def display_predictions_st(batch_texts, assistant_manager, context, batch_index):
    print("Inside of display_predictions_st")
    if 'predictions' not in st.session_state or st.session_state['new_batch']:
        print("Inside of if statement")
        predictions = process_lines(batch_texts, assistant_manager, context)
        st.session_state['predictions'] = predictions
    else:
        print("Using existing predictions")
        predictions = st.session_state['predictions']
    
    return predictions

def get_updated_csv(all_labeled_data):
    # Convert the labeled data to a DataFrame
    df = pd.DataFrame(all_labeled_data, columns=['Text', 'Label'])
    return df.to_csv(index=False).encode('utf-8')


def concatenate_and_clean(original_df, labeled_data):
    # Convert the labeled data to a DataFrame
    labeled_df = pd.DataFrame(labeled_data, columns=['Text', 'Label'])

    # Concatenate with the original DataFrame
    concatenated_df = pd.concat([original_df, labeled_df]).drop_duplicates(subset='Text', keep='last')

    return concatenated_df

# Function to offer CSV download
def offer_csv_download(original_df, all_labeled_data):
    cleaned_df = concatenate_and_clean(original_df, all_labeled_data)
    csv = cleaned_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='labeled_data.csv',
        mime='text/csv',
    )




@st.cache_resource
def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    st.title("Interactive AI Assisted Labeling")

    # File Input 
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    all_decisions_made = True 


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

    # Button to initialize the assistant
    if 'assistant_created' not in st.session_state or not st.session_state['assistant_created']:
        if st.button('Initialize Assistant'):
            print("Initializing the assistant for the first time...")
            client = OpenAI()  
            st.session_state['assistant_manager'] = assistant.OpenAIAssistantManager(client)
            st.session_state['assistant_manager'].retrieve_assistant("asst_pGF0OMtctUDdF8laiGoAiWQZ")
            st.session_state['assistant_created'] = True
    else:
        print("Assistant already created.")

    if uploaded_file is not None and st.session_state['data'] is None:
        st.session_state['data'] = load_data(uploaded_file)
    
    # Initialize session state variables if they don't exist
    if 'data' not in st.session_state:
        st.session_state['data'] = None
        st.session_state['context'] = []
        st.session_state['formatted_context'] = "" 
        st.session_state['initial_labels_submitted'] = False
        st.session_state['batch_size'] = INITIAL_BATCH_SIZE   # Define this value
        st.session_state['accuracy_scores'] = []
        st.session_state['all_labeled_data'] = []
        st.session_state['no_count'] = 0
    
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []

    if 'batch_sizes' not in st.session_state:
        st.session_state['batch_sizes'] = []

    if 'index' not in st.session_state:
        st.session_state['index'] = 0

    if 'new_batch' not in st.session_state:
        st.session_state['new_batch'] = True


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
            print("Submitted Initial Labels")
            formatted_context = update_context_with_labels(st.session_state['context'])
            st.session_state['formatted_context'] = formatted_context
            st.session_state['context'] = []  #=
            st.success('Initial Labels submitted successfully!')
            st.session_state['initial_labels_submitted'] = True

       
        if st.session_state.get('initial_labels_submitted', False):
            temp_no_count = 0
            print("in initial labels submitted loop")
            new_batch_processed = False
            batch_index = st.session_state['index'] // st.session_state['batch_size']
            batch_texts = []
            actual_batch_size = 0
            if st.session_state.get('new_batch', True) or 'predictions' not in st.session_state:
                if st.session_state['index'] < len(st.session_state['data']):

                    batch_index = st.session_state['index'] // st.session_state['batch_size']
                    batch = process_batch(st.session_state['data']['Text'].tolist(), 
                                      st.session_state['index'], 
                                      st.session_state['batch_size'], 
                                      st.session_state['batch_sizes']) 
                    batch_texts, actual_batch_size = batch

                    predictions = display_predictions_st(batch_texts, 
                                    st.session_state['assistant_manager'],
                                    st.session_state['formatted_context'], 
                                    batch_index)
                    
                    st.session_state['new_batch'] = False
                    st.session_state['batch_texts'] = batch_texts
                    st.session_state['predictions'] = predictions
                    st.session_state['new_batch_loaded'] = True  # Indicate that a new batch is loaded

                else:
                    st.write("All sentences labeled!")
                       
            # Loop to display predictions and capture user decisions
            temp_no_count = 0
            with st.container():
                print("inside of container")
                for local_index, (text, prediction) in enumerate(st.session_state['predictions']):
                    overall_index = st.session_state['index'] + local_index
                    key_suffix = f"{batch_index}_{overall_index}_{hashlib.md5(text.encode()).hexdigest()}"

                    st.write(f"Predicted for '{text}': {prediction}")
                    options = ['Yes', 'No']
                    user_decision = st.radio(f"Do you agree with this label?", options, key=f"decision_{key_suffix}", index=0)

                    if user_decision == 'No':
                        # Just create the selectbox. Streamlit will manage its state automatically.
                        print("Inside of user decision no")
                        print("no count", st.session_state['no_count'])
                        temp_no_count += 1
                        st.selectbox("Select the correct label", ['NEU', 'OTR', 'PRS', 'REP'], key=f"correct_label_{key_suffix}")
            
            if 'no_count' not in st.session_state:
                st.session_state['no_count'] = 0
            st.session_state['no_count'] += temp_no_count

            if st.button("Submit New Labels"):
                print("clicked on Submit New Labels")
                st.write("Processing predictions...")
                labeled_data, updated_context = process_user_inputs_and_prepare_next_batch(
                    st.session_state['batch_texts'], 
                    st.session_state['predictions'], 
                    st.session_state['formatted_context']
                )
                st.session_state['formatted_context'] = updated_context
                st.session_state['all_labeled_data'].extend(labeled_data)
                print("Labeled Data:", st.session_state['all_labeled_data'])

                total_predictions = len(st.session_state['predictions'])
                correct_predictions = total_predictions - st.session_state['no_count']
                accuracy = correct_predictions / total_predictions if total_predictions else 0

                print("Appending accuracy:", accuracy)
                st.session_state['accuracy_scores'].append(accuracy)  # Update accuracy_scores here

                st.session_state['no_count'] = 0

                if st.session_state.get('new_batch_loaded', False) and actual_batch_size > 0:
                    print("Appending batch size:", actual_batch_size)
                    st.session_state['batch_sizes'].append(actual_batch_size)  # Update batch_sizes here
                    st.session_state['new_batch_loaded'] = False

                if 'predictions' in st.session_state:
                    del st.session_state['predictions']

                print("Incrementing index")
                print("Index:", st.session_state['index'])
                st.session_state['index'] += st.session_state['batch_size']
                print("New index:", st.session_state['index'])

                # Display accuracy of the current batch
                st.write(f"Current Batch Accuracy: {accuracy}")


            if st.button("Keep Labeling"):
                print("clicked on Keep Labeling")
                st.session_state['new_batch'] = True
                print("New batch loaded:", st.session_state['new_batch'])
                st.write("Loading next batch. Please scroll back up to see the predictions.")
                st.rerun()

            if 'accuracy_scores' in st.session_state and 'batch_sizes' in st.session_state:
                # Data preparation
                accuracy_scores = st.session_state['accuracy_scores']
                batch_sizes = st.session_state['batch_sizes'][:-1]
                print("Accuracy Scores to save:", accuracy_scores)
                print("Batch Sizes to save:", batch_sizes)

                # Create separate DataFrames
                df_accuracy = pd.DataFrame({
                    'Batch Number': range(1, len(accuracy_scores) + 1),
                    'Accuracy': accuracy_scores
                })

                df_batch_sizes = pd.DataFrame({
                    'Batch Number': range(1, len(batch_sizes) + 1),
                    'Batch Size': batch_sizes
                })

                # Provide download buttons for CSVs
                csv_accuracy = df_accuracy.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Accuracy Data as CSV",
                    data=csv_accuracy,
                    file_name='accuracy_scores.csv',
                    mime='text/csv'
                )

                csv_batch_sizes = df_batch_sizes.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Batch Sizes Data as CSV",
                    data=csv_batch_sizes,
                    file_name='batch_sizes.csv',
                    mime='text/csv'
                )


            if st.button("Export Labeled Data"):
                # Assuming 'data' is the original DataFrame loaded from the uploaded CSV
                offer_csv_download(st.session_state['data'], st.session_state['all_labeled_data'])

if __name__ == "__main__":
    main()






