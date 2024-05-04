# Imports
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Logging Config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

loader = PyPDFLoader("my_pdf.pdf")
documents = (
    loader.load_and_split()
)  # Default chunk_size: int = 4000, chunk_overlap: int = 200,

logging.info("Document Fetched and Converted to Chunks")


# Convert to Vector and Save Vector Stores
db = FAISS.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="phi3"),
)

db.save_local("faiss_index")
logging.info("Document Chunks converted to Embeddings and Saved")
