from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Extract data from the PDF
def load_pdf(data: str):
    """
    Load all PDF files from the given folder path using PyPDFLoader.
    Example: data="data/"
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# Create text chunks
def text_split(extracted_data):
    """
    Split documents into smaller text chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download embedding model
def download_hugging_face_embeddings():
    """
    Load the HuggingFace sentence-transformer model for embeddings.
    all-MiniLM-L6-v2 => embedding dimension = 384
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings