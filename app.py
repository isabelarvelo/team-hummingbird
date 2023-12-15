# incorporates all app.py sections into one cohesive app.py file

import streamlit as st
import newest_app
import ASK_THE_DATASET
import AI_LABELING


st.set_page_config(page_title="ReTeach: AI-Assisted Coding", page_icon="🌟", layout="wide")

# header theme
st.markdown("""
    <div style="background-color:#464E5F;padding:10px;border-radius:10px;margin-bottom:10px">
    <h1 style="color:white;text-align:center;">My Streamlit App</h1>
    </div>
    """, unsafe_allow_html=True)


# main function for the streamlit app
def main():
   # st.title("ReTeach: AI-Assisted Coding")

    tab1, tab2, tab3 = st.tabs(["AI-Assisted Labeling", "Ask About the Dataset", "Interactive AI-Assisted Labeling"])

    # calling the main function from ai_assisted_labeling module, to be in tab 1
    with tab1:
        AI_LABELING.main()

    with tab2:
        ASK_THE_DATASET.main()

    with tab3:
        interactive_ai_labeling.main()
        #newest_app.main()

# Run the main function
if __name__ == "__main__":
    main()
