import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Import pipeline
from process import load_all_pdfs, process_pipeline  

# === Konfigurasi ===
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
os.makedirs(INDEX_DIR, exist_ok=True)

if __name__ == "__main__":
    print("[INFO] Memuat model embedding...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Ambil semua dokumen
    docs = load_all_pdfs()
    all_chunks = []
    metadata = []  # simpan info asal dokumen dan urutan chunk

    for name, content in docs.items():
        print(f"[PROCESS] {name}")
        chunks = process_pipeline(content, chunk_size=500, overlap=50)

        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metadata.append({"doc": name, "chunk": i, "text": c})

    print(f"[INFO] Total chunks: {len(all_chunks)}")

    # === Buat embedding ===
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    # === Buat FAISS index ===
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # === Simpan index & metadata ===
    faiss.write_index(index, os.path.join(INDEX_DIR, "docs.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"[SAVED] Index & metadata disimpan di {INDEX_DIR}")
