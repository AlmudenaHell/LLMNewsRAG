import faiss
import numpy as np

class Retriever:
    def __init__(self, embeddings, documents):
        self.documents = documents
        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def retrieve(self, query_embedding, k=2):
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )
        return [self.documents[i] for i in indices[0]]
