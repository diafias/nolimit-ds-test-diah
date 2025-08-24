# =========================
# Debug metadata
# =========================
import pickle
import faiss

# Load metadata dan index
with open(r"D:\nolimit-ds-test\index\metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

index = faiss.read_index(r"D:\nolimit-ds-test\index\docs.index")

# Fungsi retrieve
def retrieve(query, top_k=3):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    docs = [metadata[i] for i in I[0]]
    return docs

# =========================
# Debug example
# =========================
query = "ibu kota indonesia"
docs = retrieve(query, top_k=3)

print("Jumlah dokumen yang diretrieve:", len(docs))
for i, doc in enumerate(docs):
    print(f"\nDokumen ke-{i}:")
    print(doc)          # menampilkan seluruh dictionary
    print("Key yang tersedia:", list(doc.keys()))
