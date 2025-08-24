import os
import PyPDF2

# Folder tempat file PDF disimpan
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Fungsi untuk membaca teks dari satu file PDF
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Fungsi untuk membaca semua PDF dalam folder data
def load_all_pdfs(data_dir=DATA_DIR):
    docs = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(data_dir, file_name)
            print(f"[INFO] Membaca {file_name}...")
            docs[file_name] = read_pdf(file_path)
    return docs

if __name__ == "__main__":
    docs = load_all_pdfs()

    print("\n=== Hasil Baca PDF ===")
    for name, content in docs.items():
        print(f"\n--- {name} ---")
        print(content[:500])  # tampilkan 500 karakter pertama
        print("...")
