# arxiv-recommender

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://nadrajak-arxiv-recommender.streamlit.app/)

Recommender system for scientific papers from the [arXiv dataset](https://arxiv.org/help/bulk_data), based on **fine-tuned transformer embeddings** of paper abstracts.

Built with [SentenceTransformers](https://www.sbert.net/), [PyTorch](https://pytorch.org/), and [Streamlit](https://streamlit.io/).  

This repository contains:  
- Jupyter notebooks for **exploration, fine-tuning, evaluation, and recommendation**  
- A **Streamlit demo app** for paper recommendation

## Demo
Try out the **[live demo](https://nadrajak-arxiv-recommender.streamlit.app/)** on Streamlit Community Cloud.

## Setup
### Run in Google Colab
This project is designed to be easily runnable in Google Colab. Each Jupyter notebook (`.ipynb`) in the `notebooks/` directory includes a "Setup" cell that automatically handles environment setup, including mounting Google Drive, cloning the repository, and installing dependencies.

To get started, simply open and create a copy of any of the notebooks directly in Google Colab using the links below:

* **[Exploration](https://colab.research.google.com/github/nadrajak/arxiv-recommender/blob/main/notebooks/0-arxiv-exploration.ipynb)**
* **[Fine-tuning](https://colab.research.google.com/github/nadrajak/arxiv-recommender/blob/main/notebooks/1-arxiv-fine-tuning.ipynb)**
* **[Evaluation](https://colab.research.google.com/github/nadrajak/arxiv-recommender/blob/main/notebooks/2-arxiv-evaluation.ipynb)**
* **[Recommendation](https://colab.research.google.com/github/nadrajak/arxiv-recommender/blob/main/notebooks/3-arxiv-recommendation.ipynb)**


### Run locally
#### Prerequisites
- [Python 3.9+](https://www.python.org/downloads/)
        
#### Running
For local development and execution, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nadrajak/arxiv-recommender.git
    cd arxiv-recommender
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
From there, you can explore and run the notebooks, or locally host the demo using:
```bash
streamlit run app/demo.py
```
The application is then available on http://localhost:8501
