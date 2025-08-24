import pickle
import faiss
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

with open(r"D:\nolimit-ds-test-diah\index\metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

index = faiss.read_index(r"D:\nolimit-ds-test-diah\index\docs.index")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =========================
# Load LLM (google/flan-t5-base)
# =========================
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, truncation=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)  # device=-1 = CPU

def retrieve(query, top_k=2):
    # Encode query
    query_vec = embedder.encode([query], convert_to_numpy=True)
    # Search FAISS
    D, I = index.search(query_vec, top_k)
    # Ambil dokumen dari metadata
    docs = [metadata[i] for i in I[0]]
    return docs

def rag_chat(query, top_k=2, max_new_tokens=256):
    # Step 1: Retrieve documents
    docs = retrieve(query, top_k=top_k)

    # Gabungkan konteks dari field string yang benar (biasanya 'text')
    context = "\n".join([doc['text'] for doc in docs])

    # Step 2: Prompt ke LLM
    prompt = f"""
    Gunakan hanya informasi berikut untuk menjawab pertanyaan. Jangan menambahkan informasi lain. 
    Jawaban harus jelas, singkat, dan relevan.

    Konteks:
    {context}

    Pertanyaan:
    {query}

    Jawaban:"""

    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        truncation=True,
    )[0]["generated_text"]
    return result

if __name__ == "__main__":
    print("RAG Chatbot siap! (ketik 'exit' atau 'quit' untuk berhenti)")
    
    # File untuk menyimpan histori
    history_file = r"D:\nolimit-ds-test-diah\chat_history.txt"

    while True:
        q = input("User: ")
        if q.lower() in ["exit", "quit", "q"]:
            break
        try:
            answer = rag_chat(q)
            print("Bot:", answer)

            # Simpan ke file
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(f"User: {q}\n")
                f.write(f"Bot: {answer}\n")
                f.write("-" * 50 + "\n")  # separator antar pertanyaan

        except Exception as e:
            print("Terjadi error:", e)

