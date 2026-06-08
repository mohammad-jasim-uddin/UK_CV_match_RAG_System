import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Simple FAISS vector store using SentenceTransformers embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: list[str] = []

    def build_index(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("No chunks provided to build the index.")

        self.chunks = chunks
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype("float32")

        _, indices = self.index.search(query_embedding, top_k)

        results = []
        for index in indices[0]:
            if 0 <= index < len(self.chunks):
                results.append(self.chunks[index])

        return results
