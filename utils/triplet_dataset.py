import numpy as np
import pandas as pd

from datasets import Dataset as HFDataset

from sentence_transformers.evaluation import TripletEvaluator


def generate_triplets(data, n, col="abstract"):
    """Generate anchor-positive-negative triplets."""

    anchors = []
    positives = []
    negatives = []

    # Pre-compute category groups and their distribution
    category_groups = data.groupby('category')

    # Prepare helper arrays for sampling
    category_unique = np.array([])
    category_counts = np.array([])
    category_distr = np.array([])

    # Populate helper arrays & filter out categories w <2 papers (shouldn't happen)
    for cat, group in category_groups:
        cat_count = len(group)
        if cat_count >= 2:
            category_unique = np.append(category_unique, cat)
            category_counts = np.append(category_counts, cat_count)
    category_distr = category_counts / category_counts.sum()

    # Pre-compute negative category mappings
    negative_category_map = {}
    for cat in category_unique:
        negative_category_map[cat] = [c for c in category_unique if c != cat]

    for i in range(n):
        # Get anchor and positive samples
        anchor_category = np.random.choice(category_unique, p=category_distr)
        positive_group = category_groups.get_group(anchor_category)
        anchor_idx, positive_idx = np.random.choice(positive_group.index, 2, replace=False)
        
        # Get negative sample using pre-computed negative categories
        negative_category = np.random.choice(negative_category_map[anchor_category])
        negative_group = category_groups.get_group(negative_category)
        negative_idx = np.random.choice(negative_group.index)
        
        # Extract and validate texts
        anchors.append(data.loc[anchor_idx, col])
        positives.append(data.loc[positive_idx, col])
        negatives.append(data.loc[negative_idx, col])
        
    return anchors, positives, negatives


# def create_dataset_for_trainer(anchors, positives, negatives):
def create_dataset_for_trainer(data, n=None, col="abstract"):
    """Create a HuggingFace Dataset in the format expected by SentenceTransformerTrainer"""

    if n is None:
        n = len(data)

    anchors, positives, negatives = generate_triplets(data, n, col)

    # Create the dataset as a list of dictionaries
    data = []
    for anchor, positive, negative in zip(anchors, positives, negatives):
        data.append({
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        })
    
    # Convert to HuggingFace Dataset
    dataset = HFDataset.from_list(data)

    return dataset


def create_triplet_evaluator_for_trainer(dataset, name="triplet_eval"):
    """Create TripletEvaluator in the format expected by SentenceTransformerTrainer."""
    
    evaluator = TripletEvaluator(
        anchors=dataset["anchor"],
        positives=dataset["positive"],
        negatives=dataset["negative"],
        name=name
    )

    return evaluator


# def create_triplet_evaluator_for_trainer(data, n=None, col="abstract", name="triplet_eval"):
#     """Create TripletEvaluator in the format expected by SentenceTransformerTrainer."""
    
#     if n is None:
#         n = min(1000, len(data))  # Reasonable default for evaluation
    
#     anchors, positives, negatives = generate_triplets(data, n, col)
    
#     evaluator = TripletEvaluator(
#         anchors=anchors,
#         positives=positives,
#         negatives=negatives,
#         name=name
#     )

#     return evaluator
