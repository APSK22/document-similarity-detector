from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, Any
import re
from collections import Counter
import io

app = FastAPI(title="Document Similarity Detector")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize BERT model (using a smaller model for efficiency)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def preprocess_text(text: str) -> str:
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    words1 = set(preprocess_text(text1).split())
    words2 = set(preprocess_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def cosine_similarity_manual(text1: str, text2: str) -> float:
    """Calculate cosine similarity using TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0


def lsa_similarity(text1: str, text2: str, n_components: int = 100) -> float:
    """Calculate similarity using Latent Semantic Analysis."""
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Adjust n_components if needed
        n_components = min(n_components, min(tfidf_matrix.shape) - 1)
        if n_components < 1:
            return 0.0
        
        svd = TruncatedSVD(n_components=n_components)
        lsa_matrix = svd.fit_transform(tfidf_matrix)
        
        similarity = cosine_similarity(lsa_matrix[0:1], lsa_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0


def get_bert_embeddings(text: str) -> np.ndarray:
    """Get BERT embeddings for text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings


def bert_similarity(text1: str, text2: str) -> float:
    """Calculate similarity using BERT embeddings."""
    try:
        emb1 = get_bert_embeddings(text1).reshape(1, -1)
        emb2 = get_bert_embeddings(text2).reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    except:
        return 0.0


def calculate_ir_metrics(text1: str, text2: str) -> Dict[str, float]:
    """Calculate IR evaluation metrics."""
    words1 = set(preprocess_text(text1).split())
    words2 = set(preprocess_text(text2).split())
    
    if not words1 or not words2:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    
    # Treating text2 as retrieved and text1 as relevant
    true_positives = len(words1.intersection(words2))
    
    precision = true_positives / len(words2) if words2 else 0.0
    recall = true_positives / len(words1) if words1 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4)
    }


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.post("/analyze")
async def analyze_documents(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Analyze similarity between two uploaded documents.
    Files are processed in memory and not saved.
    """
    try:
        # Read files in memory
        content1 = (await file1.read()).decode('utf-8')
        content2 = (await file2.read()).decode('utf-8')
        
        if not content1.strip() or not content2.strip():
            raise HTTPException(status_code=400, detail="One or both files are empty")
        
        # Calculate all similarity metrics
        jaccard = jaccard_similarity(content1, content2)
        cosine = cosine_similarity_manual(content1, content2)
        lsa = lsa_similarity(content1, content2)
        bert = bert_similarity(content1, content2)
        
        # Calculate IR metrics
        ir_metrics = calculate_ir_metrics(content1, content2)
        
        # Get embeddings for frontend display
        vectorizer_temp = TfidfVectorizer()
        tfidf_matrix_temp = vectorizer_temp.fit_transform([content1, content2])
        feature_names = vectorizer_temp.get_feature_names_out().tolist()
        
        # Get top TF-IDF features for each document
        tfidf_doc1_array = tfidf_matrix_temp[0].toarray()[0]
        tfidf_doc2_array = tfidf_matrix_temp[1].toarray()[0]
        
        top_features_doc1 = []
        top_features_doc2 = []
        
        # Get top 15 features for each document
        top_indices_1 = np.argsort(tfidf_doc1_array)[::-1][:15]
        top_indices_2 = np.argsort(tfidf_doc2_array)[::-1][:15]
        
        for idx in top_indices_1:
            if tfidf_doc1_array[idx] > 0:
                top_features_doc1.append({
                    "feature": feature_names[idx],
                    "score": round(float(tfidf_doc1_array[idx]), 4)
                })
        
        for idx in top_indices_2:
            if tfidf_doc2_array[idx] > 0:
                top_features_doc2.append({
                    "feature": feature_names[idx],
                    "score": round(float(tfidf_doc2_array[idx]), 4)
                })
        
        # Get BERT embeddings snippet (first 10 dimensions)
        bert_emb1 = get_bert_embeddings(content1)
        bert_emb2 = get_bert_embeddings(content2)
        
        # Prepare response
        results = {
            "similarity_metrics": {
                "jaccard_index": round(jaccard, 4),
                "cosine_similarity": round(cosine, 4),
                "lsa_similarity": round(lsa, 4),
                "bert_similarity": round(bert, 4)
            },
            "ir_evaluation_metrics": ir_metrics,
            "document_info": {
                "doc1_length": len(content1),
                "doc2_length": len(content2),
                "doc1_words": len(preprocess_text(content1).split()),
                "doc2_words": len(preprocess_text(content2).split())
            },
            "embeddings": {
                "tfidf_doc1": top_features_doc1,
                "tfidf_doc2": top_features_doc2,
                "bert_dimensions": len(bert_emb1),
                "bert_doc1_sample": [round(float(x), 4) for x in bert_emb1[:10]],
                "bert_doc2_sample": [round(float(x), 4) for x in bert_emb2[:10]]
            },
            "average_similarity": round((jaccard + cosine + lsa + bert) / 4, 4)
        }
        
        return results
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Files must be text files (UTF-8 encoded)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Document Similarity Detector API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
