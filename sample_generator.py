# flake8: noqa
import os, csv
import numpy as np
import pandas as pd
import random as rand

def generate_sample_data(fname, model, tokenizer, num_samples):
    """
    Fetches and predicts scores on samples from the test data.
    """

    test_path = './train_test/toxic-comments/test'

    # Get number of test examples
    total_examples = 0
    with open(f'{test_path}/metadata.txt', 'r') as meta:
        total_examples = int(meta.read())

    # Generate random starting index
    start = rand.randint(0, total_examples - num_samples)
    stop = start + num_samples

    # Fetch text from each file
    texts = []
    for i in range(start, stop):
        with open(f'{test_path}/texts/{i}.txt') as text_f:
            texts.append(text_f.read())

    # Write predicted classes and scores to .csv file
    with open(fname, 'w') as f:
        writer = csv.Writer(f)
        writer.writerow(['Text', 'Predicted Class', 'Prediction Score', 'Highestest Toxicity Class', 'Toxicity Score'])

        for text in texts:
            pred_class, pred_score, toxic_class, toxic_score = predict(text, model, tokenizer)
            writer.writerow([text, pred_class, pred_score, toxic_class, toxic_score])


def predict(text, model, tokenizer):
    """
    Predicts likelihood of the input text for each label.
    """

    # Tokenize data
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=False
    )
    
    logits = model(**inputs).logits

    # Get label with highest score
    pred_class, pred_score = get_class_and_score(logits, model.config.id2label, 0)

    # Get label in (obscene, threat, insult, identity_hate) with highest score
    toxic_class, toxic_score = get_class_and_score(logits, model.config.id2label, 2)

    return pred_class, pred_score, toxic_class, toxic_score


def get_class_and_score(logits, id2label, start):
    """
    Returns the score and predicted class of the given results.
    """

    # start is for getting the highest toxicity score, excluding "toxic" and "severe_toxic"
    logits = logits[:, start:]
    pred_id = logits.argmax().item()

    # normalize scores using sigmoid function
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    predictions = sigmoid(logits.detach().numpy()[0])

    pred_score = predictions[pred_id]
    pred_class = id2label[pred_id + start]

    return pred_class, pred_score


def get_sample_data(fname, model, tokenizer, n):
    if not os.path.exists(fname):
        generate_sample_data(fname, model, tokenizer, n)
    return pd.read_csv(fname)