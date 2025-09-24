# src/rag_chain.py

import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

from google import genai
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# Helper: Load any document type
# -----------------------------
def load_any_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        return TextLoader(file_path).load()
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# -----------------------------
# Main RAG Class
# -----------------------------
class RAG:
    def __init__(self, persist_directory="data/vectorstore"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def build_vectorstore(self, folder_path="data/raw"):
        """Load all docs from a folder, split into chunks, and build Chroma DB."""
        all_docs = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                try:
                    docs = load_any_document(file_path)
                    all_docs.extend(docs)
                    print(f"Loaded {file}")
                except Exception as e:
                    print(f"Skipping {file}: {e}")

        if not all_docs:
            print("No valid documents found!")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)

        self.db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.db.persist()
        print(f"✅ Vector DB built with {len(chunks)} chunks from {len(os.listdir(folder_path))} files.")

    def load_vectorstore(self):
        """Load existing Chroma DB."""
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def query(self, question: str, top_k: int = 3):
        """Query the DB with similarity search and return Gemini response."""
        if not self.db:
            self.load_vectorstore()

        retriever = self.db.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.get_relevant_documents(question)

        if not docs:
            return "No relevant documents found."

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text
