# arxiv-semantic-search

Semantic search on arXiv paper abstracts from the [arXiv dataset](https://arxiv.org/help/bulk_data), using **fine-tuned transformer embeddings**. Built with the [SentenceTransformers](https://www.sbert.net/) library and the [PyTorch](https://pytorch.org/) deep learning framework.

## Demo
...


## Run in Google Colab
This project is designed to be easily runnable in Google Colab. Each Jupyter notebook (`.ipynb`) in the `notebooks/` directory includes a "Setup" cell that automatically handles environment setup, including mounting Google Drive, cloning the repository, and installing dependencies.

To get started, simply open and create a copy of any of the notebooks directly in Google Colab using the links below:

* **[0-arxiv-exploration](https://colab.research.google.com/github/nadrajak/arxiv-semantic-search/blob/main/notebooks/0-arxiv-exploration.ipynb)**
* **[1-arxiv-fine-tuning](https://colab.research.google.com/github/nadrajak/arxiv-semantic-search/blob/main/notebooks/1-arxiv-fine-tuning.ipynb)**
* **[2-arxiv-evaluation](https://colab.research.google.com/github/nadrajak/arxiv-semantic-search/blob/main/notebooks/2-arxiv-evaluation.ipynb)**
* **[3-arxiv-recommendation](https://colab.research.google.com/github/nadrajak/arxiv-semantic-search/blob/main/notebooks/3-arxiv-recommendation.ipynb)**


## Run locally
### Prerequisites
- [Python 3.9+](https://www.python.org/downloads/)
        
### Build & Run
For local development and execution, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nadrajak/arxiv-semantic-search.git
    cd arxiv-semantic-search
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
cd app
streamlit run demo.py
```
The application is then available on http://localhost:8501.
