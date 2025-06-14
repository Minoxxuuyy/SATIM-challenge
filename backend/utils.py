"""
Utility module: embeddings, Gemini, document processing, compliance scoring
"""

import os
import docx
import fitz
import numpy as np
import faiss
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

documents = []
policy_metadata = []
vector_store = None

# Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def initialize_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

gemini_model = initialize_gemini()

def load_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_pdf(path: str) -> str:
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_file(file_path: str) -> str:
    return load_docx(file_path) if file_path.endswith('.docx') else extract_text_from_pdf(file_path)

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def preprocess_policy_documents(paths: List[str]) -> Tuple[List[str], List[Dict]]:
    all_chunks, all_meta = [], []
    for path in paths:
        policy_name = os.path.splitext(os.path.basename(path))[0]
        text = load_docx(path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({"policy": policy_name, "chunk_id": i, "source": path})
    return all_chunks, all_meta

def create_vector_store(texts: List[str]) -> faiss.Index:
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index

def initialize_system(policy_paths: List[str]):
    global documents, policy_metadata, vector_store
    documents, policy_metadata = preprocess_policy_documents(policy_paths)
    vector_store = create_vector_store(documents)

def search_vector_store(query: str, k: int = 5):
    query_embedding = embedding_model.encode([query])
    distances, indices = vector_store.search(query_embedding.astype('float32'), k)
    return [(int(i), float(d)) for i, d in zip(indices[0], distances[0])]

def analyze_compliance(use_case: str, relevant_chunks: List[str]) -> Dict:
    context = "\n\n".join([f"Policy Excerpt {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
    prompt = f"""
    You are a compliance analyst. Based on the following policy excerpts and use case:
    1. Assign KPI score (0-1) per policy.
    2. Identify compromised aspects.
    3. Recommend fixes.
    Use Case:\n{use_case}\nPolicies:\n{context}
    """
    response = gemini_model.generate_content(prompt)
    return parse_gemini_response(response.text)

def parse_gemini_response(response_text: str) -> Dict:
    result = {"compliance_per_policy": [], "areas_for_improvement": []}
    lines = re.findall(r"-\s*(.+?):\s*KPI Score\s*=\s*([0-9.]+)\s*-\s*(.+)", response_text)
    for policy, score, compromise in lines:
        score = float(score)
        label = "compliant" if score >= 0.85 else "partially compliant" if score >= 0.5 else "not compliant"
        result["compliance_per_policy"].append({
            "policy": policy.strip(), "kpi_score": score,
            "compromised": compromise.strip(), "compliance_label": label
        })
    if "Recommendations:" in response_text:
        recs = re.findall(r"-\s*(.+)", response_text.split("Recommendations:")[-1])
        result["areas_for_improvement"] = [r.strip() for r in recs]
    return result

def segment_text_into_clauses(text: str) -> List[str]:
    return [line.strip() for line in text.split('\n') if len(line.strip()) > 50]

def embed_clauses(clauses: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(clauses, convert_to_tensor=True)

def compare_clauses(user_clauses: List[str], standard_clauses: List[str], model, threshold=0.75):
    u_emb = embed_clauses(user_clauses, model)
    s_emb = embed_clauses(standard_clauses, model)
    results = []
    for i, uc in enumerate(user_clauses):
        sims = util.cos_sim(u_emb[i], s_emb)[0]
        max_score, max_index = float(sims.max()), int(sims.argmax())
        if max_score >= threshold:
            results.append({
                "user_clause": uc,
                "matched_standard_clause": standard_clauses[max_index],
                "similarity": round(max_score, 3)
            })
    return results

def check_coverage_from_standard(standard_clauses, user_clauses, model, threshold=0.75):
    s_emb = embed_clauses(standard_clauses, model)
    u_emb = embed_clauses(user_clauses, model)
    matched, missing = [], []
    for i, sc in enumerate(standard_clauses):
        sims = util.cos_sim(s_emb[i], u_emb)[0]
        max_score = float(sims.max())
        if max_score >= threshold:
            matched.append({
                "standard_clause": sc,
                "matched_user_clause": user_clauses[int(sims.argmax())],
                "similarity": round(max_score, 3)
            })
        else:
            missing.append({
                "standard_clause": sc,
                "max_similarity": round(max_score, 3)
            })
    return matched, missing

def filter_relevant_clauses(clauses, model, anchor_texts=None, threshold=0.4):
    if not anchor_texts:
        anchor_texts = [
            "rules and regulations", "data privacy", "safety procedures",
            "intellectual property", "compliance requirements"
        ]
    clause_emb = embed_clauses(clauses, model)
    anchor_emb = embed_clauses(anchor_texts, model)
    return [clauses[i] for i, emb in enumerate(clause_emb)
            if float(util.cos_sim(emb, anchor_emb)[0].max()) >= threshold]