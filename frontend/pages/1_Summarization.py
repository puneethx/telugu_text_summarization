# frontend/pages/1_Chat_With_LLM.py
import streamlit as st
from model_ import ModelInf
import os

# Load model instance
CHECKPOINT_PATH = r"D:\Capstone\finetune_checkpoint\check_16050_2.2765516408210438_.pth"
CSV_PATH = r"C:\Users\punee\OneDrive\Desktop\Yolo env\capstone\jsontocsv\output.csv"
model = ModelInf(CHECKPOINT_PATH, csv_path=CSV_PATH)

# Page setup
st.set_page_config(page_title="Chat with LLM", page_icon="💬")
st.title("Chat with the Summarizer 💬")

# Input text
telugu_paragraph = st.text_area("Enter a Telugu paragraph to summarize:")

# Summarize or fetch headline
if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):
        result = model.summarize_text(telugu_paragraph)
        if result:
            st.success("Summary generated successfully!")
            st.markdown(f"**Summary/Headline:**\n{result}")
        else:
            st.error("Failed to generate summary.")
