import numpy as np
import pandas as pd


def normalize_whitespace(data, text_columns=["authors", "title", "abstract"]):
    """Normalize whitespace in specified text columns of a DataFrame."""

    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].fillna("").astype(str) 
            data[col] = data[col].str.replace(r"[\r\n\t]+", " ", regex=True) 
            data[col] = data[col].str.replace(r"\s+", " ", regex=True)      
            data[col] = data[col].str.strip()
    
    return data


def normalize_abstracts(data, min_length=30, max_length=300):
    """Filter abstracts by word count to ensure quality."""

    data["word_count"] = data["abstract"].str.split().str.len()
    
    data = data[data["word_count"].between(min_length, max_length)]
    data = data.drop(columns=["word_count"])

    return data


def truncate_categories(data):
    """Simplify and normalize category labels in the DataFrame."""
    
    # Extract broad categories
    data["categories"] = data["categories"].str.split(" ").str[0]  # Keep only first category
    data["categories"] = data["categories"].str.split(".").str[0]  # Throw away subcategories 
    
    # Normalize categories with common prefixes
    data["categories"] = np.where(data["categories"].str.startswith("hep"), "hep", data["categories"])
    data["categories"] = np.where(data["categories"].str.startswith("nucl"), "nucl", data["categories"])

    # Add mathematical physics to physics
    data["categories"] = np.where(data["categories"].str.startswith("math-ph"), "physics", data["categories"])

    data = data.rename(columns={"categories": "category"})
    
    return data
