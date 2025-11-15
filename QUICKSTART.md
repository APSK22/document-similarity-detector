# Document Similarity Detector - Quick Start Guide

## 🚀 Running the Application

### Option 1: Using the start script (Recommended)
```bash
./start.sh
```

### Option 2: Manual start
```bash
source venv/bin/activate
python main.py
```

Then open: http://localhost:8000

---

## 📓 Running the Jupyter Notebook

### Quick Start (Recommended)
```bash
./run_notebook.sh
```

### Manual Start
```bash
source venv/bin/activate
jupyter notebook document_similarity_analysis.ipynb
```

### First Time Setup (if not already done)
```bash
source venv/bin/activate
pip install jupyter ipykernel pandas
pip install --only-binary :all: matplotlib seaborn
```

**Note**: Using `--only-binary :all:` avoids compilation issues with Python 3.14

The notebook includes:
- ✅ All similarity algorithms with explanations
- ✅ Interactive visualizations
- ✅ **Collapsible embedding displays**
- ✅ Step-by-step code walkthrough

---

## 🎯 Features Overview

### Web Application Features:
1. **Upload two documents** (text files)
2. **View similarity scores**:
   - Jaccard Index
   - Cosine Similarity (TF-IDF)
   - LSA Similarity
   - BERT Similarity
3. **IR Metrics**: Precision, Recall, F1-Score
4. **Interactive charts** and visualizations
5. **Collapsible embedding displays**:
   - TF-IDF vectors with top features
   - BERT embedding samples
   - Feature importance bars

### Jupyter Notebook Features:
- Complete implementation walkthroughs
- Mathematical explanations
- Visualization of embeddings
- Collapsible HTML displays for vectors
- Comprehensive analysis dashboard

---

## 📂 Sample Documents

Use the provided sample documents to test:
- `sample_doc1.txt` - AI and Machine Learning
- `sample_doc2.txt` - Similar content with variations

Expected similarity: ~0.65-0.85 (documents are semantically similar)

---

## 🔧 Troubleshooting

### Server won't start
- Make sure virtual environment is activated
- Check if port 8000 is available
- First run takes longer (BERT model download ~90MB)

### Import errors
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Jupyter not found
```bash
pip install jupyter ipykernel
```

---

## 📊 Understanding the Results

### Similarity Scores (0.0 - 1.0):
- **0.0 - 0.3**: Low similarity (different topics)
- **0.3 - 0.6**: Moderate similarity (related topics)
- **0.6 - 0.8**: High similarity (similar content)
- **0.8 - 1.0**: Very high similarity (nearly identical)

### IR Metrics:
- **Precision**: How many retrieved terms are relevant
- **Recall**: How many relevant terms are retrieved
- **F1-Score**: Harmonic mean of precision and recall

---

## 🎓 Educational Value

This project demonstrates key IRSW concepts:
- ✅ Vector Space Models
- ✅ TF-IDF Weighting
- ✅ Cosine Similarity
- ✅ Dimensionality Reduction (LSA)
- ✅ Neural Embeddings (BERT)
- ✅ Information Retrieval Evaluation

Perfect for learning and demonstrating IR/Semantic Web principles!
