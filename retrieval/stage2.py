import os
import faiss
import pickle
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# -------------------------
# Setup Embedding Model
# -------------------------
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# -------------------------
# Fetch Data from FineWeb API
# -------------------------
def fetch_fineweb(query: str, k: int = 5) -> List[Dict]:
    """
    Fetch documents from FineWeb API for a given query.
    Returns list of dicts: [{"doc_id": ..., "text": ...}, ...]
    """
    url = "https://clueweb22.us/fineweb/search"
    params = {"query": query, "k": k}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("⚠️ Error fetching from FineWeb:", response.status_code)
        return []

    results = response.json().get("results", [])
    docs = []
    for i, item in enumerate(results):
        try:
            # Decode Base64 JSON
            import base64, json
            decoded = json.loads(base64.b64decode(item).decode("utf-8"))
            text = decoded.get("text", "")
        except Exception:
            text = str(item)

        docs.append({"doc_id": f"fineweb_{query}_{i}", "text": text})

    return docs


# -------------------------
# Chunking Helper
# -------------------------
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# -------------------------
# Build FAISS Index
# -------------------------
def build_faiss_index(documents: List[Dict], save_path="faiss_index.bin", meta_path="metadata.pkl"):
    """
    Build FAISS index from documents and save index + metadata.
    """
    all_chunks, metadata = [], {}
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for pos, chunk in enumerate(chunks):
            idx = len(all_chunks)
            all_chunks.append(chunk)
            metadata[idx] = {
                "doc_id": doc["doc_id"],
                "chunk_text": chunk,
                "position": pos
            }

    # Encode chunks
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, save_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Indexed {len(all_chunks)} chunks from {len(documents)} docs")
    return save_path, meta_path


# -------------------------
# Load FAISS Index
# -------------------------
def load_faiss_index(save_path="faiss_index.bin", meta_path="metadata.pkl"):
    index = faiss.read_index(save_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# -------------------------
# Search FAISS Index
# -------------------------
def search_faiss(sub_query: str, index, metadata, top_k: int = 3):
    """
    Search FAISS index for one sub-query.
    Returns top chunks with metadata.
    """
    vec = embedder.encode([sub_query], convert_to_numpy=True)
    D, I = index.search(vec, top_k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append({
            "text": meta["chunk_text"],
            "doc_id": meta["doc_id"],
            "position": meta["position"],
            "score": float(dist)
        })
    return results

# -------------------------
# Stage 2 Assembly
# -------------------------
def stage2(stage1_output: Dict, top_k: int = 3):
    """
    Full Stage 2 pipeline:
    1. Fetch FineWeb docs for each sub-query
    2. Build FAISS index
    3. Retrieve relevant chunks for each sub-query
    """
    query = stage1_output["query"]
    sub_queries = stage1_output["sub_questions"]

    # Step 1: Fetch documents
    documents = []
    for sq in sub_queries:
        documents.extend(fetch_fineweb(sq, k=3))

    if not documents:
        print("⚠️ No documents retrieved from FineWeb")
        return {}

    # Step 2: Build FAISS
    build_faiss_index(documents)

    # Step 3: Load FAISS
    index, metadata = load_faiss_index()

    # Step 4: Retrieve chunks
    archival_context = []
    for sq in sub_queries:
        chunks = search_faiss(sq, index, metadata, top_k=top_k)
        archival_context.append({"sub_query": sq, "chunks": chunks})

    return {"archival_context": archival_context}