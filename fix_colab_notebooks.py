import json
import glob
import sys


if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
    print("Usage: python fix_notebooks.py [directory]")
    print("Fixes metadata in Colab Jupyter notebooks for proper GitHub rendering")
    sys.exit(0)

# Find all notebooks in the directory
# Use a command-line argument for the directory, or default to the current directory
directory = sys.argv[1] if len(sys.argv) > 1 else '.'
notebook_files = glob.glob(f'{directory}/*.ipynb')

if not notebook_files:
    print(f"No notebook files (.ipynb) found in the directory: {directory}")
    sys.exit(0)

print(f"Processing {len(notebook_files)} notebook(s)...")

for filepath in notebook_files:
    try:
        # Open and read the notebook
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Remove widgets metadata (causes rendering issues)
        nb['metadata'].pop('widgets', None)

        # Ensure pygments_lexer is set for proper cell background rendering
        if 'language_info' not in nb['metadata']:
            nb['metadata']['language_info'] = {}
        if 'pygments_lexer' not in nb['metadata']['language_info']:
            nb['metadata']['language_info']['pygments_lexer'] = 'ipython3'
        
        # Write the changes back to the same file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
            f.write('\n')

        print(f"  Successfully processed: {filepath}")

    except Exception as e:
        print(f"Failed to process {filepath}: {e}")

print("All notebooks have been processed.")
