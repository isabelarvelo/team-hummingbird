# incorporates all app.py sections into one cohesive app.py file

import streamlit as st
import newest_app
import ASK_THE_DATASET
import AI_LABELING
import getpass
import os
from openai import OpenAI



os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
client = OpenAI()

# set page configuration
st.set_page_config(page_title="ReTeach: AI-Assisted Coding", page_icon="🌟", layout="wide")

# function to apply local CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply CSS
local_css("style.css")

# custom header
st.markdown("""
    <div style="background-color:#464E5F;padding:10px;border-radius:10px;margin-bottom:10px">
    <h1 style="color:white;text-align:center;">ReTeach: AI-Assisted Coding</h1>
    </div>
    """, unsafe_allow_html=True)

# main function for the streamlit app
def main():

    tab1, tab2, tab3 = st.tabs(["AI-Assisted Labeling", "Ask About the Dataset", "Interactive AI-Assisted Labeling"])

    # calling the main function from ai_assisted_labeling module, to be in tab 1
    with tab1:
        AI_LABELING.main()

    with tab2:
        ASK_THE_DATASET.main()

    with tab3:
        newest_app.main()

# Run the main function
if __name__ == "__main__":
    main()
