#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Start the FastAPI server
echo "Starting Document Similarity Detector..."
echo "Server will be available at: http://localhost:8000"
echo "API Documentation at: http://localhost:8000/docs"
echo ""
echo "Note: First startup may take 30-60 seconds while loading BERT model..."
echo ""

python main.py
