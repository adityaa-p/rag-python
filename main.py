from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

file_path = "/Users/adityapalpattuwar/Downloads/python-handbook.pdf"
loader = PyPDFLoader(file_path)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

#Vector embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = QdrantVectorStore.from_documents(documents=texts, embedding=embedding_model, url="http://localhost:6333", collection_name="test_collection")
