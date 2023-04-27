# flake8: noqa
import os
import pandas as pd
import streamlit as st
from sample_generator import get_sample_data, predict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)


def fetch_model(cache_dir):
    """
    Fetches the model and tokenizer from the cache directory
    """
    tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    config = AutoConfig.from_pretrained(cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(cache_dir, config=config)
    return tokenizer, model


def streamlit_app(model, tokenizer, model_dict, sample):
    # Create two tabs
    welc, sent = st.tabs(["Welcome", "Sentiment Analysis"])

    welc.title("👋 Welcome to TTSA!")
    welc.markdown(
        """
        TTSA (Toxic Tweet Sentiment Analysis) is a BERT-based language learning model that classifies the toxicity of a text input. Select the **Sentiment Analysis** tab to try it out!

        ### About
        TTSA performs multi-label classification with six labels $-$ **Toxic**, **Severe** **Toxic**, **Obscene**, **Threat**, **Insult**, & **Identity Hate**. The Obscene, Threat, Insult, & Identity Hate labels are treated as the four types of toxicity.

        For each input, TTSA assigns a score (0 to 1) for each of the labels. The label with the highest score is the *Predicted Class* and the highest-scoring toxicity type is the *Highest Toxicity Class*.
        
        ### Resources Used
        Pretrained Base Model $-$ [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert#overview)  
        Data Set $-$ [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

        ### Source Code
        - [Github Repo](https://github.com/TomY-Zhang/Toxic-Tweets-Language-Model)
        - [Hugging Face Space](https://huggingface.co/spaces/TomYZhang/toxic-tweets/tree/main)
        """
    )

    sent.title("Toxic Tweet Sentiment Analysis")
    selected_model = sent.selectbox('Select a learning model', ('distilbert-base-uncased',))

    # Form for user to enter text
    with sent.form("Sample Form"):
        text = st.text_input("Enter text here", "What are you, stupid?")

        # When submitted, the model predicts scores for the input text
        submitted = st.form_submit_button("Analyze")
        if submitted:
            tokenizer = model_dict[selected_model]["tokenizer"]
            model = model_dict[selected_model]["model"]
            pred_class, pred_score, toxic_class, toxic_score = predict(text, model, tokenizer)

            # Display the results in a table
            st.write(pd.DataFrame({
                'Predicted class': [pred_class,],
                'Prediction Score': [pred_score,],
                'Highest Toxicity Class': [toxic_class,],
                'Toxicity Score': [toxic_score,],
            }))
    
    sent.markdown("## Sample Data")
    os.environ["sample_size"] = str(sent.slider("Rows to Display", min_value=10, max_value=len(sample), value=10, step=1))

    sample_size = os.environ.get("sample_size")
    sample_size = int(sample_size) if sample_size is not None else 10

    # Display a table of sample data with the desired number of rows
    sent.write(sample.sample(n=sample_size).reset_index(drop=True))


def main():
    cache_dir = './saved_model'
    tokenizer, model = fetch_model(cache_dir)

    model_dict = {
        'distilbert-base-uncased' : {
            "tokenizer" : tokenizer,
            "model" : model
        }
    }

    fname = "sample.csv"
    sample = get_sample_data(fname, model, tokenizer, 200)

    streamlit_app(model, tokenizer, model_dict, sample)


if __name__ == '__main__':
    main()

