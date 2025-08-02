import os
from utils import load_pdf, split_documents, get_embedding_model, store_embeddings

file_path = os.getenv("PDF_FILE")
chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
vector_db_url = os.getenv("VECTOR_DB_URL")
collection_name = os.getenv("COLLECTION_NAME")

print("🔍 Loading PDF...")
documents = load_pdf(file_path)

print("✂️ Splitting into chunks...")
texts = split_documents(documents, chunk_size, chunk_overlap)

print(f"🧠 Creating embeddings and storing to Qdrant collection: {collection_name}")
embedding_model = get_embedding_model()
store_embeddings(texts, embedding_model, collection_name, vector_db_url)
