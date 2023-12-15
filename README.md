# ai-assisted-coding-final

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

## Motivation

Manually labeling classroom transcript data is expensive, slow, and
tedious. Our goal was to create an internal tool to help accelerate this
process for our subject matter experts that “learns” while the user is
labeling and iteratively creates better and better predictions.

Specifically, our colleagues requested:

- Something to help them label transcript data faster
- A model to provide label guesses to review
- The ability to ask questions about the data

We met these needs by providing:

- Interactive pages for assisted labeling
- In-the-loop learning during manual labeling
- Question answering directly using the transcript data

## Install

``` sh
pip install ai_assisted_coding_final
```

## Jupyter Notebooks

### AI Labeling

This notebook uses HuggingFace’s DeBERTa-v3-base fine-tuned zero-shot
classification pipelin to add label suggestions to transcript CSV files.

Key features:

- Uploads a transcript CSV file
- Passes each row to a DistilBERT zero-shot classification model
- Returns labels and confidences for each row
- Creates a new DataFrame with original text, suggested labels, and
  confidences
- Styles the DataFrame to color code confidences
- Allows saving the output CSV with predictions

### Ask the Dataset

This notebook uses OpenAI’s Chat Completion API to answer questions
about transcript CSV files.

Key features:

- Accepts CSV file uploads
- Concatenates multiple files into one string
- Takes a natural language question string as input
- Queries the ChatGPT API using the context + question
- Returns the API response with the answer
- Provides examples of asking and answering questions

### In-the-loop AI-Assisted Labeling

This notebook interatively improves labeling predictions using user
feedback.

Key features:

- Shows batches of rows and label predictions
- Allows correcting predictions and retraining on feedback
- Keeps accuracy metrics for each batch
- Increase batch size as accuracy improves
- Saves labeled CSV with user feedback incorporated

### Interactive Labeling

This module provides functions for interactive labeling of text data
with AI assistance and user validation.

Key features:

- Load text data from a CSV file
- Send text to an AI assistant to predict labels
- Prompt user to manually enter a label
- Convert labeled data to a Pandas DataFrame
- Increase batch size if accuracy is high enough
- Get a batch of data from the unlabeled set
- Show predictions to user to validate and collect labels
- Determine accuracy versus user-provided labels

The workflow enabled is:

- Load unlabeled text data
- Process batches with AI predictions
- Show predictions to user to validate/fix
- Increase batch size if accuracy is high enough
- Repeat until all data is labeled
- Save final labeled dataset

This allows iterative improvement of labeling accuracy with a
human-in-the-loop. The `interactive-labeling-demo` notebook in this repo
is a demo for this module’s functionality.

## User Interface

### AI-Assisted Labeling Page

This page allows users to get label suggestions from a model.

Key features:

- Upload a transcript CSV file
- Display the file as a stylized DataFrame with predictions
- Download a CSV with predictions added

### Ask the Dataset Page

This page enables asking natural language questions about transcripts.

Key features:

- Select one or more transcript CSVs
- Enter a text question about the data
- Submit question and view response
- Ask multiple independent or follow-up questions and see chat history

### AI-Assisted Labeling Page

This page interatively improves labeling with user feedback.

Key features:

- Pick files to label
- Label statements in batches
- Retrains on user-fixed labels after each batch
- Increases batch size as accuracy improves
- Tracks metrics like accuracy per batch
- Save final labeled CSV

## Running the App

To run the Streamlit app:

1.  Ensure Docker is installed
2.  Open this repository in a dev container
3.  Run streamlit run app.py

Note: For several functionalities, OpenAI API keys are required. Have
your open AI key prepared.

## Data

Csv files processed for this use case will contain is a recording of a
classroom session with two columns - one for the transcribed text, and
another for the labels. Some csvs may only have one column as there are
no labels yet provided. Each row is a statement made by the teacher. The
labels are:

- **OTR**: An opportunity to respond (e.g., ‘What color is this candy?’)
- **PRS**: Praise (e.g., “Great job, Morgan!”)
- **REP**: Reprimand (e.g., “You need to keep quiet while someone else
  is reading.”)
- **NEU**: None of the above

## Team

This project was completed by:

- Isabel Arvelo
- Jackie Himel
- Holly Hou
- Sankhya Sivakumar
