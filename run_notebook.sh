#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Launch Jupyter Notebook
echo "🚀 Launching Jupyter Notebook..."
echo "Opening document_similarity_analysis.ipynb"
echo ""
echo "The notebook will open in your browser automatically."
echo "Press Ctrl+C to stop the Jupyter server when done."
echo ""

jupyter notebook document_similarity_analysis.ipynb
