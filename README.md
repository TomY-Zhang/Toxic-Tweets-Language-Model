---
title: Toxic Tweets Language Model
emoji: 📚
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---

## Hugging Face Space
- [Hugging Face Space URL](https://huggingface.co/spaces/TomYZhang/toxic-tweets)

## Pretrained Models
- [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert#overview)

## Data Set
- [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Requirements

- `Docker`
- `python 3.9`
- `pandas==2.0.1`
- `scikit-learn==1.0.2`
- `streamlit==1.21.0`
- `torch==2.0.0`
- `transformers==4.27.4`

## Docker Setup
1. Navigate to the [Docker website](https://www.docker.com/products/docker-desktop/) and install the latest version of Docker Desktop compatible with your operating system.

2. To ensure that Docker is properly installed, open your terminal and execute `docker version`.<br> This will display the properties of Docker version installed.

3. Download the Pytorch image from [Docker Hub](https://hub.docker.com/r/pytorch/pytorch).

4. To build the Docker image, execute `docker build -t streamlit .` in the terminal.

5. Execute `docker run -p 8501:8501 streamlit` to create a container from the `streamlit` image.

6. View the Streamlit app at http://0.0.0.0:8501

## Deploying the App
1. Execute `docker build -t streamlit .` in the terminal to build a Docker image named `streamlit`.

2. Execute `docker run -p 8501:8501 streamlit` to create a container from the `streamlit` image. The webapp can be accessed at http://0.0.0.0:8501.
