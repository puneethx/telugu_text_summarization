import streamlit as st

st.set_page_config(
    page_title="Telugu Summarization ",
    page_icon="💬",
)

st.title("Telugu Summarization Using Deep Learning💬")

st.markdown(
    """
    Abstract:
    In the era of information overload, summarization plays a pivotal role in extracting meaningful insights from large text corpora. 

This project presents a cutting-edge Telugu Dialogue Summarization Model designed to generate concise and coherent summaries from conversational data in Telugu.

Leveraging the XLSTM architecture, the model is trained on an extensive Telugu corpus, fine-tuned with the TESUM dataset, and optimized for linguistic nuances using advanced tokenization techniques, including the GPT-4 O Tokenizer.

By incorporating start and end tokens, we enhanced the model's ability to understand context and generate precise summaries, offering a valuable tool for applications such as media analysis, customer support, and conversational AI in regional languages.
    """
)

# st.image("./img/streamlit_cover_img.png")