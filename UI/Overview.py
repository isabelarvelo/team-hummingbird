import streamlit as st
from PIL import Image


def main():
    st.title("Welcome to ReTeach! :wave: ")

    # Display an image 
    # Ensure the image file is in the same directory as your script, or provide the full path.
    image = Image.open('GPT_LOGO.png')
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image(image, caption='Enhancing Teaching with Technology', width=300)

    st.markdown("""
        ## Empowering Educators with Data-Driven Insights

        **ReTeach** is a revolutionary platform that listens to classroom interactions 
        and provides real-time, AI-powered feedback to educators. Our goal is to 
        enhance teaching effectiveness and improve learning outcomes.

        ### How ReTeach Works:
        - **Upload Transcripts:** Teachers upload their classroom interaction transcripts.
        - **AI Analysis:** Our AI model analyzes the interactions and suggests improvements.
        - **Feedback & Growth:** Teachers receive personalized feedback for professional development.

        Explore the sidebar to navigate through the application's features and start your journey towards enhanced educational experiences!
    """)

if __name__ == "__main__":
    main()



