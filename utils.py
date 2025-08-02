import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = file_path
        doc.metadata["page_label"] = doc.metadata.get("page", "Unknown")
    return documents

def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def get_embedding_model():
    return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

def store_embeddings(texts, embedding_model, collection_name, url):
    return QdrantVectorStore.from_documents(
        documents=texts,
        embedding=embedding_model,
        url=url,
        collection_name=collection_name
    )

def get_vector_store(embedding_model, collection_name, url):
    return QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name=collection_name,
        url=url
    )