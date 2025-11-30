from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

# Load data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Upload text chunks
PineconeVectorStore.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)

print("✅ Data uploaded to Pinecone index:", PINECONE_INDEX_NAME)