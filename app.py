# flake8: noqa
import streamlit as st
from transformers import AutoTokenizer, RobertaForSequenceClassification

def main():
    st.title("Text Sentiment Analysis")

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    labels = model.config.id2label

    with st.form("Sample Form"):
        text = st.text_input("Enter text here", "I love the weather today!")
        submitted = st.form_submit_button("Submit")

        if submitted:
            inputs = tokenizer(text, return_tensors="pt")
            sentiments = {}
            logits = model(**inputs).logits.tolist()[0]
            for i, val in enumerate(logits):
                sentiments[labels[i]] = val
            st.write(sentiments)
                

if __name__ == '__main__':
    main()
