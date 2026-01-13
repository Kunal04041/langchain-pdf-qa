import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self):
        # Using a reliable local embedding model (free, open source)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.store = None

    def create_store(self, documents):
        """
        Builds a FAISS index from documents.
        """
        self.store = FAISS.from_documents(documents, self.embeddings)
        return self.store

    def similarity_search(self, query, k=3):
        """
        Retrieves relevant documents for a query.
        """
        if not self.store:
            return []
        return self.store.similarity_search(query, k=k)

    def save(self, path="faiss_index"):
        if self.store:
            self.store.save_local(path)

    def load(self, path="faiss_index"):
        if os.path.exists(path):
            self.store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            return True
        return False
