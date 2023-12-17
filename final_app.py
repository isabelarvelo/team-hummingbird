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
    if st.button('Initialize Assistant', key='initialize_assistant'):
        client = OpenAI()
        st.session_state['assistant_manager'] = assistant.OpenAIAssistantManager(client)
        st.session_state['assistant_manager'].retrieve_assistant("asst_pGF0OMtctUDdF8laiGoAiWQZ")

@st.cache_resource
def load_data(file_path):
    return pd.read_csv(file_path)

# submits the initial labels and handles button clicks
def submit_initial_labels():
    if st.button('Submit Initial Labels', key='submit_initial_labels_button'):
        formatted_context = update_context_with_labels(st.session_state['context'])
        st.session_state['formatted_context'] = formatted_context
        st.session_state['context'] = []  # Clear context after submission
        st.success('Initial Labels submitted successfully!')
        st.session_state['initial_labels_submitted'] = True

        # Directly call display_predictions_st to generate predictions for the next batch
        start_index = st.session_state['index']
        end_index = min(start_index + INITIAL_BATCH_SIZE, len(st.session_state['data']))
        batch_texts = st.session_state['data']['Text'].iloc[start_index:end_index].tolist()
        if batch_texts:
            st.session_state['predictions'] = display_predictions_st(batch_texts, st.session_state['assistant_manager'], formatted_context, start_index, labeled_data=None)
            st.session_state['index'] = end_index
            st.session_state['new_batch'] = False


# updating batch size based on accuracy - for first round, increase by 5 if accuracy above 80%
# for second round, increase by 5 if accuracy above 90%
# for third round (and all rounds after) increase by 5 if accuracy is 100%
# def adjust_batch_size(accuracy_scores, current_batch_size):
#     cumulative_accuracy = sum(accuracy_scores) / len(accuracy_scores)
#     thresholds = {1: 0.8, 2: 0.9}
#     batch_sizes = {1: 10, 2: 15}
#
#     for threshold_batch, accuracy_threshold in thresholds.items():
#         if len(accuracy_scores) == threshold_batch and cumulative_accuracy > accuracy_threshold:
#             return batch_sizes[threshold_batch]
#     return current_batch_size

def adjust_batch_size(accuracy_scores, current_batch_size):
    if not accuracy_scores:  # Check if the list is empty
        return current_batch_size  # Return the current batch size if there are no accuracy scores

    cumulative_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    thresholds = {1: 0.8, 2: 0.9}
    batch_sizes = {1: 10, 2: 15}

    for threshold_batch, accuracy_threshold in thresholds.items():
        if len(accuracy_scores) == threshold_batch and cumulative_accuracy > accuracy_threshold:
            return batch_sizes[threshold_batch]
    return current_batch_size


# updated context for labeling (for assistant)
def update_context_with_labels(context_items):
    context_str = "Example(s) of how to classify utterances:"
    for text, label in context_items:
        context_str += f"\nuser: '{text}'\nassistant: '{label}'"
    context_str += "\nOnly answer with one the following three-letter labels: OTR, PRS, REP, NEU"
    return context_str


def process_user_inputs_and_prepare_next_batch(batch_texts, predictions, context):
    labeled_data = []
    for index, (text, original_prediction) in enumerate(zip(batch_texts, predictions)):
        key_suffix = f"{index}_{hashlib.md5(text.encode()).hexdigest()}"
        user_decision = st.session_state.get(f"decision_{key_suffix}", 'Yes')
        selected_label = st.session_state.get(f"correct_label_{key_suffix}", original_prediction[1])
        correct_label = selected_label if user_decision == 'No' else original_prediction[1]
        labeled_data.append((text, correct_label))

    updated_context = update_context_with_labels(labeled_data)
    return labeled_data, updated_context

def display_predictions():
    st.markdown("<h3 style='font-weight: bold;'>Predictions:</h3>", unsafe_allow_html=True)
    if 'predictions' in st.session_state:
        for i, (text, prediction) in enumerate(st.session_state['predictions']):
            st.write(f"Text: {text}")
            key_suffix = f"{st.session_state['prediction_session_id']}_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
            new_label_key = f"new_label_{key_suffix}"
            current_label = st.session_state.get(new_label_key, prediction)
            edited_label = st.selectbox("Choose a new label", ["OTR", "PRS", "REP", "NEU"], key=new_label_key, index=["OTR", "PRS", "REP", "NEU"].index(current_label))
            st.session_state['predictions'][i] = (text, edited_label)
            st.write("---")

        # Global Save Changes button
        # if st.button("Save Changes"):
        #     # Update the predictions with the edited labels
        #     for i, (text, _) in enumerate(st.session_state['predictions']):
        #         new_label_key = f"new_label_{i}"
        #         if new_label_key in edited_labels:
        #             st.session_state['predictions'][i] = (text, edited_labels[new_label_key])
        #     st.success("Changes saved successfully!")


def display_predictions_st(batch_texts, assistant_manager, context, batch_index, labeled_data):
    updated_context = context

    if labeled_data is not None:
        for text, label in labeled_data:
            updated_context += f"\nuser: '{text}'\nassistant: '{label}'"

    predictions = process_lines(batch_texts, assistant_manager, updated_context)
    st.session_state['predictions'] = predictions
    return st.session_state['predictions']


# initializing session states - setting default values
def initialize_session_state():

    st.session_state['all_labeled_data'] = []

    # Initialize session state variables if they don't exist
    if 'data' not in st.session_state:
        st.session_state['data'] = None
        st.session_state['context'] = []

        st.session_state['formatted_context'] = ""
        st.session_state['initial_labels_submitted'] = False
        st.session_state['batch_size'] = INITIAL_BATCH_SIZE   # Define this value
        st.session_state['accuracy_scores'] = []
       # st.session_state['all_labeled_data'] = []
        st.session_state['no_count'] = 0


    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []

    if 'batch_sizes' not in st.session_state:
        st.session_state['batch_sizes'] = []

    if 'index' not in st.session_state:
        st.session_state['index'] = 0

    if 'new_batch' not in st.session_state:
        st.session_state['new_batch'] = True

    if 'prediction_session_id' not in st.session_state:
        st.session_state['prediction_session_id'] = 0

    if 'widget_counter' not in st.session_state:
        st.session_state['widget_counter'] = 0

def process_lines(lines, assistant_manager, context=""):


    print("inside process lines")
    print("lines:", lines)
    data = []
    additional_context = "Return a list of labels for each utterance. Each utterance is separated by \n"
    context += additional_context
    assistant_manager.create_thread(context)

    all_lines = "\n ".join(lines)

    try:
        completed_run = assistant_manager.submit_message(all_lines)
        response_page = assistant_manager.get_response()
        messages = [msg for msg in response_page]
        assistant_message = messages[-1].content[0].text.value
        print("assistant message:", assistant_message)
        labels = assistant_message.replace('\n', ' ').replace(',', ' ').split()

        # Check if labels are one of the specified labels
        valid_labels = ["NEU", "OTR", "PRS", "REP"]
        labels = [label if label in valid_labels else "NEU" for label in labels]
        print("labels:", labels)

    except Exception as e:
        print(f"An error occurred: {e}")
        labels = ["NEU"] * len(lines)

    data = list(zip(lines, labels))
    print("data:", data)
    return(data)


def display_labeling_interface():
    # Check if 'data' exists in the session state and has been loaded
    if 'data' in st.session_state and st.session_state['data'] is not None and st.session_state['index'] < len(st.session_state['data']):
        current_data = st.session_state['data'].iloc[st.session_state['index']]
        text = current_data['Text']
        st.markdown("<h3 style='font-weight: bold;'>Sentence to Label:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='border: 2px solid #ccc; background-color: #F5F5DC; padding: 10px; text-align: center; font-size: 18px;'>{text}</div><br>", unsafe_allow_html=True)

        label_options = ["OTR", "PRS", "REP", "NEU"]
        cols = st.columns(len(label_options))

        for i, option in enumerate(label_options):
            # Create a unique key for each button
            unique_key = f"label_button_{st.session_state['index']}_{option}_{i}"
            if cols[i].button(f"{option}", key=unique_key):
                st.session_state['context'].append((text, option))
                st.session_state['index'] += 1
                break

# updated function to enable user to continue labeling after first batch of predictions
def save_and_continue_labeling():
    # Ensure start_index and end_index are integers
    start_index = int(st.session_state['index'])
    end_index = int(min(start_index + int(st.session_state['batch_size']), len(st.session_state['data'])))

    # Process submitted labels and update context
    if 'predictions' in st.session_state:
        labeled_data = [(text, label) for text, label in st.session_state['predictions']]
        updated_context = update_context_with_labels(labeled_data)
        st.session_state['formatted_context'] = updated_context
        st.session_state['all_labeled_data'].extend(labeled_data)

        # Calculate accuracy and update batch size
        total_predictions = len(st.session_state['predictions'])
        correct_predictions = sum(1 for (text, prediction), user_label in zip(st.session_state['predictions'], labeled_data) if prediction == user_label[1])
        accuracy = correct_predictions / total_predictions if total_predictions else 0
        new_batch_size = adjust_batch_size(st.session_state['accuracy_scores'], accuracy)
        st.session_state['batch_size'] = new_batch_size
        st.write(f"Updated batch size for next iteration: {new_batch_size}")
        st.session_state['accuracy_scores'].append(accuracy)

    # Generate predictions for the next batch
    if start_index < len(st.session_state['data']):
        batch_texts = st.session_state['data']['Text'].iloc[start_index:end_index].tolist()
        if batch_texts:
            predictions = display_predictions_st(batch_texts, st.session_state['assistant_manager'], updated_context, start_index, labeled_data=None)
            st.session_state['predictions'] = predictions
            st.session_state['index'] = end_index
        else:
            st.write("No more texts to process.")
    else:
        st.write("All sentences labeled!")


# processing submitted labels, returns accuracy of predictions, updates batch size, and updates accuracy score
def process_submitted_labels():
    if st.button("Submit New Labels", key='submit_new_labels_button'):

        # Collect all labels from the current predictions
        labeled_data = [(text, label) for text, label in st.session_state['predictions']]

        # Update the context with the new labels
        updated_context = update_context_with_labels(labeled_data)
        st.session_state['formatted_context'] = updated_context

        # Extend the all_labeled_data with the new labeled data
        if 'all_labeled_data' not in st.session_state:
            st.session_state['all_labeled_data'] = []
        st.session_state['all_labeled_data'].extend(labeled_data)
        st.session_state['new_batch'] = True

        # Calculate accuracy
        total_predictions = len(st.session_state['predictions'])
        correct_predictions = sum(1 for (text, prediction), user_label in zip(st.session_state['predictions'], labeled_data) if prediction == user_label[1])
        accuracy = correct_predictions / total_predictions if total_predictions else 0

        # Update batch size
        new_batch_size = adjust_batch_size(st.session_state['accuracy_scores'], accuracy)
        st.session_state['batch_size'] = new_batch_size
        st.write(f"Updated batch size for next iteration: {new_batch_size}")

        if new_batch_size != st.session_state['batch_size']:
            st.write("Batch size updated due to high accuracy in predictions.")
        st.session_state['accuracy_scores'].append(accuracy)  # Update accuracy scores

        # Move to the next batch
        st.session_state['index'] += st.session_state['batch_size']

# def process_batch_predictions():
#     print("processing batch predictions")
#
#     if st.session_state.get('initial_labels_submitted', False) and st.session_state.get('new_batch', True):
#         total_data = len(st.session_state['data'])
#         start_index = st.session_state['index']
#         end_index = start_index + st.session_state['batch_size']
#
#         if start_index < total_data:
#             batch_texts = st.session_state['data']['Text'].iloc[start_index:end_index].tolist()
#             st.session_state['batch_texts'] = batch_texts
#
#             if batch_texts:
#                 predictions = display_predictions_st(batch_texts, st.session_state['assistant_manager'], st.session_state['formatted_context'], start_index, labeled_data=None)
#                 st.session_state['predictions'] = predictions
#                 st.session_state['index'] += st.session_state['batch_size']
#                 st.session_state['new_batch'] = False
#             else:
#                 print("No batch texts to process")
#         else:
#             st.write("All sentences labeled!")

def process_batch_predictions():
    print("processing batch predictions")

    if st.session_state.get('initial_labels_submitted', False) and st.session_state.get('new_batch', True):
        total_data = len(st.session_state['data'])
        start_index = st.session_state['index']
        end_index = start_index + st.session_state['batch_size']

        if start_index < total_data:
            batch_texts = st.session_state['data']['Text'].iloc[start_index:end_index].tolist()
            st.session_state['batch_texts'] = batch_texts

            if batch_texts:
                predictions = display_predictions_st(batch_texts, st.session_state['assistant_manager'], st.session_state['formatted_context'], start_index, labeled_data=None)
                st.session_state['predictions'] = predictions
                st.session_state['index'] += st.session_state['batch_size']
                st.session_state['new_batch'] = False
            else:
                print("No batch texts to process")
                st.session_state['new_batch'] = False  # Ensure this is set to False to avoid loop
        else:
            st.write("All sentences labeled!")
            st.session_state['initial_labels_submitted'] = False  # Reset this to handle new data if needed



# below are all the functions for handling exporting data
def offer_csv_download(df, label, file_name, key_suffix):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=label,
        data=csv,
        file_name=file_name,
        mime='text/csv',
        key=f'download_button_{key_suffix}'  # Unique key for each download button
    )

# Function to export labeled data
def export_labeled_data():
    if st.button("Export Labeled Data", key='export_labeled_data_button'):
        # Debugging: Print the current all_labeled_data
        st.write("Current labeled data:", st.session_state.get('all_labeled_data', []))

        if st.session_state.get('all_labeled_data'):
            # Ensure that the data is in the expected format
            labeled_df = pd.DataFrame(st.session_state['all_labeled_data'], columns=['Text', 'Label'])
            if not labeled_df.empty:
                csv = labeled_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Labeled Data as CSV",
                    data=csv,
                    file_name="labeled_data.csv",
                    mime='text/csv'
                )
            else:
                st.error("Labeled data is empty.")
        else:
            st.error("No labeled data to export.")


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

        offer_csv_download(df_accuracy, "Download Accuracy Data as CSV", "accuracy_data.csv", "accuracy")
        offer_csv_download(df_batch_sizes, "Download Batch Sizes Data as CSV", "batch_sizes_data.csv", "batch_sizes")

# displays labeled sentence and enables editing of label
def display_and_edit_labeled_sentences():
    st.markdown("<h3 style='font-weight: bold;'>Labeled Sentences:</h3>", unsafe_allow_html=True)

    if 'context' in st.session_state and st.session_state['context']:
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
    else:
        st.write("No labeled sentences to display.")


def main():
    initialize_session_state()
    st.title("Interactive AI-Assisted Labeling")
    initialize_assistant()

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_file_uploader")

    if uploaded_file is not None and st.session_state['data'] is None:
        st.session_state['data'] = load_data(uploaded_file)
        st.session_state['index'] = 0
        st.session_state['initial_labels_submitted'] = False

    display_labeling_interface()
    display_and_edit_labeled_sentences()

    if not st.session_state.get('initial_labels_submitted', False):
        submit_initial_labels()

    # Display predictions if initial labels are submitted and it's not a new batch
    if st.session_state.get('initial_labels_submitted', False) and not st.session_state.get('new_batch', False):
        display_predictions()

    # 'Save and Continue Labeling' button
    if st.session_state.get('initial_labels_submitted', False) and st.button("Save and Continue Labeling", key='save_continue_labeling_button'):
        save_and_continue_labeling()
        display_predictions()  # Display the new predictions after updating

    export_labeled_data()
    export_accuracy_and_batch_sizes()

if __name__ == "__main__":
    main()

