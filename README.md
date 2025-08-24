# NoLimit DS Test — RAG Chatbot Indonesia 🇮🇩

## 📌 Deskripsi
Proyek ini adalah implementasi **Retrieval-Augmented Generation (RAG) Chatbot** untuk Bahasa Indonesia.  
Chatbot ini menggunakan **FAISS** sebagai vector database untuk menyimpan embedding teks, serta **Hugging Face Transformers** untuk pemodelan NLP.  

Tujuan utama:  
- Membuat sistem tanya-jawab berbasis dokumen (document QA).  
- Melatih dan mengevaluasi pipeline RAG Chatbot menggunakan dataset Bahasa Indonesia.  
- Menguji pemahaman kandidat terhadap NLP, embedding, dan integrasi model Hugging Face.

---

## 🚀 Fitur
- Load dan preprocessing dokumen (PDF/TXT).  
- Chunking teks dengan overlap.  
- Membuat embedding dengan Hugging Face model.  
- Menyimpan embedding ke **FAISS**.  
- Query ke FAISS untuk mengambil dokumen yang relevan.  
- Menggunakan LLM (Hugging Face model) untuk menjawab pertanyaan.  

---

## 📂 Dataset

Dokumen PDF yang digunakan sebagai sumber pengetahuan:
- Ekonomi Indonesia - Wikipedia bahasa Indonesia, ensiklopedia bebas.pdf
- Indonesia - Wikipedia bahasa Indonesia, ensiklopedia bebas.pdf
- Sejarah Indonesia - Wikipedia bahasa Indonesia, ensiklopedia bebas.pdf
📖 Sumber data: Wikipedia Bahasa Indonesia

## ⚙️ Setup Instructions

1. **Clone Repository**
```
git clone https://github.com/<username>/nolimit-ds-test-<name>.git
cd nolimit-ds-test-<name>
```
2. **Buat Virtual Environment**
```
python -m venv env-nolimit
source env-nolimit/bin/activate   # Linux/Mac
env-nolimit\Scripts\activate     # Windows
```
3. **Install Dependencies**
```
pip install -r requirements.txt
```

## 🧠 Model & Tools
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (Hugging Face)
- Vector Index: FAISS
- LLM (Generator): google/flan-t5-base (Hugging Face Transformers)
- Document Loader: PyPDF2
- Text Splitter: LangChain RecursiveCharacterTextSplitter

## 🚀 Workflow
1. Load & Clean Data (load_doc.py, process.py)
- Membaca PDF
- Preprocessing: cleaning, stopwords removal, lemmatization
- Chunking teks

2. Build Index (build_index.py)
- Embed tiap chunk menggunakan all-MiniLM-L6-v2
- Simpan FAISS index (docs.index) dan metadata (metadata.pkl)

3. RAG Chat (rag_chat.py)
- Retrieve top-k dokumen relevan dari FAISS
- LLM (flan-t5-base) menghasilkan jawaban berdasarkan konteks

## ▶️ Cara Menjalankan
```
python src/rag_chat.py
```

## 💬 Contoh Interaksi
```
RAG Chatbot siap! (ketik 'exit' atau 'quit' untuk berhenti)
User: Apa ibu kota Indonesia?
Bot: Ibu kota Indonesia adalah Jakarta. 
```

## 📌 Pipeline
Ditulis secara terpisah pada file workflow.pdf
