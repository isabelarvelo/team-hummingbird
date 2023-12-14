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

# initializing assistant
def initialize_assistant():
    if 'assistant_created' not in st.session_state or not st.session_state['assistant_created']:
        if st.button('Initialize Assistant'):
            client = OpenAI()  # Ensure you have OpenAI imported
            st.session_state['assistant_manager'] = assistant.OpenAIAssistantManager(client)
            st.session_state['assistant_manager'].retrieve_assistant("asst_pGF0OMtctUDdF8laiGoAiWQZ")
            st.session_state['assistant_created'] = True

@st.cache_resource
def load_data(file_path):
    return pd.read_csv(file_path)

# submits the initial labels and handles button clicks
def submit_initial_labels():
    if st.button('Submit Initial Labels'):
        formatted_context = update_context_with_labels(st.session_state['context'])
        st.session_state['formatted_context'] = formatted_context
        st.session_state['context'] = []  # Clear context after submission
        st.success('Initial Labels submitted successfully!')
        st.session_state['initial_labels_submitted'] = True
        st.session_state['new_batch'] = True


    # updating batch size based on accuracy - for first round, increase by 5 if accuracy above 80%
# for second round, increase by 5 if accuracy above 90%
# for third round (and all rounds after) increase by 5 if accuracy is 100%
def adjust_batch_size(current_batch_size, accuracy):
    if current_batch_size == 5 and accuracy > 0.80:
        return 10
    elif current_batch_size < 20 and accuracy > 0.90:
        return current_batch_size + 5
    elif current_batch_size >= 20 and accuracy == 1.00:
        return current_batch_size + 5
    return current_batch_size

# updated context for labeling (for asssitant)
def update_context_with_labels(context_items):
    context_str = "Example(s) of how to classify utterances:"
    for text, label in context_items:
        context_str += f"\nuser: '{text}'\nassistant: '{label}'"
    context_str += "\nOnly answer with one the following three-letter labels: OTR, PRS, REP, NEU"
    return context_str

# processes user input and prepares next batch of text for prediction
def process_user_inputs_and_prepare_next_batch(batch_texts, predictions, context):
    labeled_data = []
    for index, (text, original_prediction) in enumerate(zip(batch_texts, predictions)):
        key_suffix = f"{index}_{hashlib.md5(text.encode()).hexdigest()}"
        user_decision = st.session_state.get(f"decision_{key_suffix}", 'Yes')
        selected_label = st.session_state.get(f"correct_label_{key_suffix}", original_prediction[1])
        correct_label = selected_label if user_decision == 'No' else original_prediction[1]
        labeled_data.append((text, correct_label))


# displaying predictions for batches
def display_predictions_st(batch_texts, assistant_manager, context, batch_index):
    if 'predictions' not in st.session_state or st.session_state['new_batch']:
        predictions = process_lines(batch_texts, assistant_manager, context)
        st.session_state['predictions'] = predictions
    else:
        predictions = st.session_state['predictions']
    return predictions


    # # Update context
    # for text, label in labeled_data:
    #     context += f"\nuser: '{text}'\nassistant: '{label}'"
    #
    # return labeled_data, context


# initializing session states - setting default values
def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state['data'] = None
        st.session_state['context'] = []
        st.session_state['formatted_context'] = ""
        st.session_state['initial_labels_submitted'] = False
        st.session_state['batch_size'] = INITIAL_BATCH_SIZE
        st.session_state['accuracy_scores'] = []
        st.session_state['all_labeled_data'] = []
        st.session_state['no_count'] = 0

    # if 'index' not in st.session_state:
    #     st.session_state['index'] = 0

# original displaying predictions
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



# labeling interface for text data aka sentences
def display_labeling_interface():
    if st.session_state['data'] is not None and st.session_state['index'] < len(st.session_state['data']):
        current_data = st.session_state['data'].iloc[st.session_state['index']]
        text = current_data['Text']
        st.markdown("<h3 style='font-weight: bold;'>Sentence to Label:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='border: 2px solid #ccc; background-color: #F5F5DC; padding: 10px; text-align: center; font-size: 18px;'>{text}</div><br>", unsafe_allow_html=True)

        label_options = ["OTR", "PRS", "REP", "NEU"]
        cols = st.columns(len(label_options))

        for i, option in enumerate(label_options):
            unique_key = f"label_button_{st.session_state['index']}_{option}"
            if cols[i].button(f"{option}", key=unique_key):
                st.session_state['context'].append((text, option))
                st.session_state['index'] += 1
                break


            # processing submitted labels
#     """
#     Processes the submitted labels by performing the following steps:
#     - If the "Submit New Labels" button is clicked, it retrieves the batch texts, predictions, and formatted context from the session state.
#     - It calls the "process_user_inputs_and_prepare_next_batch()" function to process the user inputs and prepare the next batch of data.
#     - It updates the formatted context in the session state.
#     - It extends the list of all labeled data with the labeled data obtained from the previous step.
#     - It calculates the accuracy by comparing the predictions with the user labels.
#     - It adjusts the batch size based on the calculated accuracy.
#     - It updates the batch size in the session state.
#     - It displays the updated batch size for the next iteration.
#     - If the batch size is updated due to high accuracy, it displays a message indicating the update.
#     - It appends the accuracy score to the list of accuracy scores in the session state.
#     - It resets the no count variable in the session state.
#     """
def process_submitted_labels():
    if st.button("Submit New Labels"):
        labeled_data, updated_context = process_user_inputs_and_prepare_next_batch(
            st.session_state['batch_texts'],
            st.session_state['predictions'],
            st.session_state['formatted_context']
        )
        st.session_state['formatted_context'] = updated_context
        st.session_state['all_labeled_data'].extend(labeled_data)

        # Calculate accuracy
        total_predictions = len(st.session_state['predictions'])
        correct_predictions = sum(1 for (text, prediction), user_label in zip(st.session_state['predictions'], labeled_data) if prediction == user_label[1])
        accuracy = correct_predictions / total_predictions if total_predictions else 0

        # Update batch size
        new_batch_size = adjust_batch_size(st.session_state['batch_size'], accuracy)
        st.session_state['batch_size'] = new_batch_size
        st.write(f"Updated batch size for next iteration: {new_batch_size}")

        if new_batch_size != st.session_state['batch_size']:
            st.write("Batch size updated due to high accuracy in predictions.")
        st.session_state['accuracy_scores'].append(accuracy)  # Update accuracy scores
        st.session_state['no_count'] = 0

# processing predictions per batch
def process_batch_predictions():
    if st.session_state.get('initial_labels_submitted', False) and (st.session_state.get('new_batch', True) or 'predictions' not in st.session_state):
        if st.session_state['index'] < len(st.session_state['data']):
            batch_texts = st.session_state['data']['Text'].tolist()[st.session_state['index']:st.session_state['index'] + st.session_state['batch_size']]
            batch_index = st.session_state['index'] // st.session_state['batch_size']
            predictions = display_predictions_st(batch_texts,
                                                 st.session_state['assistant_manager'],
                                                 st.session_state['formatted_context'],
                                                 batch_index)  # Add batch_index here
            st.session_state['new_batch'] = False
            st.session_state['batch_texts'] = batch_texts
            st.session_state['predictions'] = predictions
        else:
            st.write("All sentences labeled!")

# below are all the functions for handling exporting data
def offer_csv_download(df, key_suffix):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'labeled_data_{key_suffix}.csv',
        mime='text/csv',
        key=f'download_button_{key_suffix}'  # Unique key for each download button
    )

def export_labeled_data():
    if st.button("Export Labeled Data"):
        if st.session_state.get('all_labeled_data'):
            labeled_df = pd.DataFrame(st.session_state['all_labeled_data'], columns=['Text', 'Label'])
            offer_csv_download(labeled_df, 'labeled_data')

def export_accuracy_and_batch_sizes():
    if 'accuracy_scores' in st.session_state and 'batch_sizes' in st.session_state:
        df_accuracy = pd.DataFrame({
            'Batch Number': range(1, len(st.session_state['accuracy_scores']) + 1),
            'Accuracy': st.session_state['accuracy_scores']
        })
        df_batch_sizes = pd.DataFrame({
            'Batch Number': range(1, len(st.session_state['batch_sizes']) + 1),
            'Batch Size': st.session_state['batch_sizes']
        })
        # st.markdown("Download Accuracy Data:")
        # offer_csv_download(df_accuracy)
        # st.markdown("Download Batch Sizes Data:")
        # offer_csv_download(df_batch_sizes)

        offer_csv_download(df_accuracy, 'accuracy')
        offer_csv_download(df_batch_sizes, 'batch_sizes')


# displays labeled sentence and enables editing of label
def display_and_edit_labeled_sentences():
    st.markdown("<h3 style='font-weight: bold;'>Labeled Sentences:</h3>", unsafe_allow_html=True)
    for i, (sentence, label) in enumerate(st.session_state['context']):
        sentence_hash = hashlib.md5(sentence.encode()).hexdigest()[:8]
        edit_key = f"edit_{i}_{sentence_hash}"
        new_label_key = f"new_label_{i}_{sentence_hash}"

        st.markdown(f"<div style='display: flex; align-items: center; justify-content: space-between;'>"
                    f"<span style='flex-grow: 2;'>{sentence} [{label}]</span>"
                    f"<button style='margin-left: 10px;' onclick='document.getElementById(\"{new_label_key}\").style.display = \"block\";'>Edit</button>"
                    f"</div>", unsafe_allow_html=True)

        if st.button("Save", key=edit_key):
            new_label = st.session_state.get(new_label_key)
            if new_label:
                st.session_state['context'][i] = (sentence, new_label)

        new_label = st.selectbox("Choose a new label", ["OTR", "PRS", "REP", "NEU"], key=new_label_key, index=["OTR", "PRS", "REP", "NEU"].index(label))
        st.markdown(f"<div id='{new_label_key}' style='display: none;'>{new_label}</div>", unsafe_allow_html=True)


def main():

    initialize_session_state()
    st.title("Interactive AI-Assisted Labeling")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None and st.session_state['data'] is None:
        st.session_state['data'] = load_data(uploaded_file)
        st.session_state['index'] = 0

    initialize_assistant()
    display_labeling_interface()
    display_and_edit_labeled_sentences()
    submit_initial_labels()
    process_batch_predictions()


    if st.session_state.get('initial_labels_submitted', False):
        process_submitted_labels()

    export_labeled_data()
    export_accuracy_and_batch_sizes()

  # if st.button("Submit New Labels"):
  #               print("clicked on Submit New Labels")
  #               st.write("Processing predictions...")
  #               labeled_data, updated_context = process_user_inputs_and_prepare_next_batch(
  #                   st.session_state['batch_texts'],
  #                   st.session_state['predictions'],
  #                   st.session_state['formatted_context']
  #               )
  #               st.session_state['formatted_context'] = updated_context
  #               st.session_state['all_labeled_data'].extend(labeled_data)
  #               print("Labeled Data:", st.session_state['all_labeled_data'])
  #
  #               # UPDATED BATCH PREDICTION LOGIC
  #               total_predictions = len(st.session_state['predictions'])
  #               correct_predictions = sum(1 for (text, prediction), user_label in zip(st.session_state['predictions'], labeled_data) if prediction == user_label[1])
  #               accuracy = correct_predictions / total_predictions if total_predictions else 0
  #
  #               # After calculating accuracy in the "Submit New Labels" button section
  #               new_batch_size = adjust_batch_size(st.session_state['batch_size'], accuracy)
  #               st.session_state['batch_size'] = new_batch_size
  #
  #               st.write(f"Updated batch size for next iteration: {new_batch_size}")
  #               if new_batch_size != st.session_state['batch_size']:
  #                   st.write("Batch size updated due to high accuracy in predictions.")
  #

if __name__ == "__main__":
    main()
