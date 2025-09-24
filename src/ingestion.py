import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "data/vectorstore"

def _get_chroma_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    # Use new API
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client

def ingest_csv_to_chroma(csv_path: str, text_column: str, id_column: str):
    df = pd.read_csv(csv_path)
    client = _get_chroma_client()

    # Create (or get) a collection
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="docs", embedding_function=embedding_function
    )

    # Prepare data
    ids = df[id_column].astype(str).tolist()
    texts = df[text_column].astype(str).tolist()

    # Add to Chroma
    collection.add(documents=texts, ids=ids)
    print(f"Ingested {len(ids)} documents into Chroma at {CHROMA_DIR}")
