import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Konfigurasi
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")

def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "docs.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search(query, top_k=3):
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = model.encode([query])
    
    index, metadata = load_index()
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "doc": metadata[idx]["doc"],
            "chunk": metadata[idx]["chunk"],
            "text": metadata[idx]["text"],
            "score": float(dist),
        })
    return results

if __name__ == "__main__":
    while True:
        q = input("\nTanya: ")
        if q.lower() in ["exit", "quit"]:
            break
        results = search(q, top_k=3)
        print("\n[HASIL PENCARIAN]")
        for r in results:
            print(f"- {r['doc']} | Chunk {r['chunk']} (score {r['score']:.4f})")
            print(r["text"][:300], "...")
