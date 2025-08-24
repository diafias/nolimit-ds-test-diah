import os
from typing import List
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import fungsi load PDF
from load_doc import load_all_pdfs  

# === STEP 1: CLEANING & NORMALIZATION ===
def clean_text(text: str) -> str:
    """
    Membersihkan teks dasar:
    - Lowercase
    - Hilangkan karakter spesial & angka
    - Hilangkan extra whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # hanya huruf
    text = re.sub(r"\s+", " ", text).strip()
    return text


# === STEP 2: TOKENIZATION, STOPWORD REMOVAL, LEMMATIZATION ===
def preprocess_text(text: str) -> str:
    """
    Preprocessing dasar:
    - Tokenisasi
    - Stopword removal
    - Lemmatization
    Output berupa string lagi (bukan list)
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("indonesian") + stopwords.words("english"))

    tokens = nltk.word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(tokens)


# === STEP 3: CHUNKING (LangChain) ===
def chunk_text_lc(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Membagi teks menjadi potongan menggunakan LangChain RecursiveCharacterTextSplitter
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # <--- ubah ke 300
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return text_splitter.split_text(text)

# === STEP 4: PIPELINE FINAL ===
def process_pipeline(raw_text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    cleaned = clean_text(raw_text)
    preprocessed = preprocess_text(cleaned)
    chunks = chunk_text_lc(preprocessed, chunk_size=chunk_size, overlap=overlap)
    return chunks


# === MAIN ===
if __name__ == "__main__":
    # Ambil semua dokumen dari folder data/
    docs = load_all_pdfs()

    all_chunks = {}
    for name, content in docs.items():
        print(f"\n[PROCESS] {name}")
        chunks = process_pipeline(content, chunk_size=500, overlap=50)
        all_chunks[name] = chunks

        print(f"Jumlah chunk: {len(chunks)}")
        for i, c in enumerate(chunks[:3]):  # tampilkan 3 chunk pertama aja
            print(f"--- {name} | Chunk {i+1} ---")
            print(c[:300])
            print("...")

    # (opsional) simpan hasil preprocessing ke file txt
    out_dir = os.path.join(os.path.dirname(__file__), "..", "processed")
    os.makedirs(out_dir, exist_ok=True)

    for name, chunks in all_chunks.items():
        base = os.path.splitext(name)[0]
        out_path = os.path.join(out_dir, f"{base}_chunks.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(c + "\n\n")
        print(f"[SAVED] {out_path}")
