RANDOM_SEED = 123

FT_NROWS = 15_000
FT_TRAIN_TRIPLETS = 10_000
FT_EVAL_TRIPLETS = 1_000
FT_TEST_TRIPLETS = 1_000


import matplotlib.pyplot as plt

def setup_plot_style():
    """Configure matplotlib styling for consistent plots across the project."""
    plt.style.use('bmh')
    
    # Custom modifications
    plt.rcParams.update({
        'font.size': 12,
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#f5f5f5',
        'axes.titlepad': 12,
        'axes.labelpad': 12
    })
    