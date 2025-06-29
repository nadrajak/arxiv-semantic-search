import kagglehub
import os


def load_kaggle_dataset(dataset_name, dataset_file):
    """Download a dataset from Kaggle and return its file path."""

    dataset_dir = kagglehub.dataset_download(dataset_name)
    dataset_path = os.path.join(dataset_dir, dataset_file)
    
    return dataset_path
    

def load_arxiv_dataset():
    """Load the arXiv dataset from Kaggle."""
    
    return load_kaggle_dataset("Cornell-University/arxiv", "arxiv-metadata-oai-snapshot.json")
