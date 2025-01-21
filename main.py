from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env.local')

# Initialize Qdrant Cloud client
client = QdrantClient(
    url=os.environ.get("QDRANT_ENDPOINT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)

collection_name = "risk"

# Check if the collection exists
existing_collections = client.get_collections().collections
if any(collection.name == collection_name for collection in existing_collections):
    print(f"Collection '{collection_name}' already exists.")
    # Delete the existing collection
    client.delete_collection(collection_name)
    print(f"Deleted existing collection '{collection_name}'.")

# Create the collection with the correct vector size
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=1024,  # Correct vector size (for your specific model)
        distance=models.Distance.COSINE
    )
)
print(f"Collection '{collection_name}' created successfully.")

# Get collection info
collection_info = client.get_collection(collection_name=collection_name)
print(f"Collection '{collection_name}' details:")
print(f"Vector size: {collection_info.config.params.vectors.size}")
print(f"Distance metric: {collection_info.config.params.vectors.distance}")

# Load the document
loader = PyPDFLoader("risk.pdf")
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
texts = text_splitter.split_documents(docs)

# Check device configuration for embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Configure embedding model
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding model loaded successfully.")

# Embed the text and set up Qdrant index
print("Embedding the text and setting up Qdrant index...")

try:
    qdrant = Qdrant.from_documents(
        documents=texts,
        embedding=embeddings,
        url=os.environ.get("QDRANT_ENDPOINT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=False,
        collection_name=collection_name,
    )
    print("Qdrant Cloud indexing created successfully.")
except Exception as e:
    print(f"An error occurred while creating the Qdrant index: {e}")
