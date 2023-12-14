import streamlit as st
from transformers import pipeline
import numpy as np
import pandas as pd
from ai_assisted_coding_final.Ai_assisted_labeling import *


# Main function for the Streamlit app
def main():
    st.title("AI-Assisted Labeling :pencil2:")
    st.write("""
        Welcome to the AI-Assisted Labeling page!

        This tool is designed to enhance your data labeling workflow. Follow these simple steps:
        
        1. Upload Your Data: Use the upload button below to upload a CSV file containing your classroom transcripts. 
        2. Review the Labels: Once uploaded, our AI model will automatically suggest labels for your data. These labels are highlighted for easy identification.
        3. Download Labeled Data: After reviewing the AI-suggested labels, you can download the labeled file. 
        
        :clock1: Please wait while we process your file. This can take 30 seconds to 3 minutes based on the size of your file.
    """)

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Legal' in df.columns:
            df.drop('Legal', axis=1, inplace=True)

        # Initialize the zero-shot classifier
        classifier = pipeline("zero-shot-classification", model="sileod/deberta-v3-base-tasksource-nli")
        # Define candidate labels
        candidate_labels = ['PRS', 'REP', 'OTR', 'NEU']
        # Initialize columns for scores
        score_columns = ['PRS_Score', 'REP_Score', 'OTR_Score', 'NEU_Score']
        for col in score_columns:
            df[col] = 0.0
            
        # Process each text and apply classifier and override rules
        for index, row in df.iterrows():
            text = row['Text']
            # Run classifier
            prediction = classifier(text, candidate_labels, truncation=True, max_length=1024)
            label_scores = {label: score for label, score in zip(prediction['labels'], prediction['scores'])}

            # Apply rule-based override using the method from the class instance
            override_label = apply_rule_based_override(text)
            if override_label:
                label_scores[override_label] = max(label_scores[override_label], 0.5)  # Override score if higher

             # Update the DataFrame with scores
            for label in candidate_labels:
                df.at[index, f'{label}_Score'] = label_scores[label]
        
        # determine the label with the highest score
        df['Label'] = df[score_columns].idxmax(axis=1).str.replace('_Score', '')
        new_column_order = [col for col in df.columns if col not in score_columns and col != 'Label'] + score_columns + ['Label']
        df = df[new_column_order]

        # Apply the styling
        score_columns = ['PRS_Score', 'REP_Score', 'OTR_Score', 'NEU_Score']
        #styled_df = df.style.applymap(color_map, subset=score_columns)
        styled_df = df.style.map(color_maps_for_st, subset=score_columns)

        st.write(styled_df)

        # Create a link for downloading the processed DataFrame
        st.markdown(get_table_download_link(df, score_columns), unsafe_allow_html=True)
if __name__ == "__main__":
    main()

