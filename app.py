# incorporates all app.py sections into one cohesive app.py file
import streamlit as st
import newest_app
import ASK_THE_DATASET
import AI_LABELING
import getpass
import os
from openai import OpenAI
import overview as overview_app
from streamlit_option_menu import option_menu



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


# Sidebar navigation
with st.sidebar:
    selected = option_menu("Main Menu",["Overview", "AI-Assisted Labeling", "Ask About the Dataset", "Interactive AI-Assisted Labeling"], menu_icon="cast", default_index=0)
    
if selected=="Overview":
    overview_app.main()
elif selected=="AI-Assisted Labeling":
    AI_LABELING.main()
elif selected=="Ask About the Dataset":
    ASK_THE_DATASET.main()
elif selected=="Interactive AI-Assisted Labeling":
    newest_app.main()


