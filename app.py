# flake8: noqa
import os, csv
import numpy as np
import pandas as pd
import random as rand
import streamlit as st
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

def generate_sample_data(fname, model, tokenizer, num_samples):
    test_path = './train_test/toxic-comments/test'

    total_examples = 0
    with open(f'{test_path}/metadata.txt', 'r') as meta:
        total_examples = int(meta.read())

    start = rand.randint(0, total_examples - num_samples)
    stop = start + num_samples

    texts = []
    for i in range(start, stop):
        with open(f'{test_path}/texts/{i}.txt') as text_f:
            texts.append(text_f.read())

    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Text', 'Predicted Class', 'Prediction Score', 'Toxicity Type', 'Toxicity Score'])

        for text in texts:
            pred_class, pred_score, toxic_class, toxic_score = predict(text, model, tokenizer)
            writer.writerow([text, pred_class, pred_score, toxic_class, toxic_score])


def predict(text, model, tokenizer):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=False
    )
    
    logits = model(**inputs).logits
    pred_class, pred_score = get_class_and_score(logits, model.config.id2label, 0)
    toxic_class, toxic_score = get_class_and_score(logits, model.config.id2label, 2)

    return pred_class, pred_score, toxic_class, toxic_score


def get_class_and_score(logits, id2label, start):
    logits = logits[:, start:]
    pred_id = logits.argmax().item()

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    predictions = sigmoid(logits.detach().numpy()[0])

    pred_score = predictions[pred_id]
    pred_class = id2label[pred_id + start]

    return pred_class, pred_score


def main():
    cache_dir = './saved_model'
    tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    config = AutoConfig.from_pretrained(cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(cache_dir, config=config)

    models = {
        'distilbert-base-uncased' : {
            "tokenizer" : tokenizer,
            "model" : model
        }
    }

    fname = "sample.csv"
    if not os.path.exists(fname):
        generate_sample_data(fname, model, tokenizer, 50)
    
    sample_data = pd.read_csv(fname)

    st.title("Toxic Comment Sentiment Analyzer")
    selected_model = st.selectbox('Select a learning model', ('distilbert-base-uncased',))

    with st.form("Sample Form"):
        text = st.text_input("Enter text here", "What are you, stupid?")

        submitted = st.form_submit_button("Submit")
        if submitted:
            tokenizer = models[selected_model]["tokenizer"]
            model = models[selected_model]["model"]
            pred_class, pred_score, toxic_class, toxic_score = predict(text, model, tokenizer)
            
            st.write(pd.DataFrame({
                'Predicted class': pred_class,
                'Prediction Score': pred_score,
                'Toxicity Class': toxic_class,
                'Toxicity Score': toxic_score
            }))
    
    st.subheader("Sample observations and predictions")
    st.dataframe(sample_data)

if __name__ == '__main__':
    main()
