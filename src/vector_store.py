import chromadb
from pathlib import Path


class VectorStoreManager:
    def __init__(self, persist_directory="vectorstore"):
        """
        Initialize a persistent ChromaDB vector database.
        """

        self.persist_directory = str(Path(persist_directory))

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="research_documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents, embeddings):
        """
        Store document chunks and their embeddings.

        Args:
            documents: List of dictionaries containing text and metadata.
            embeddings: List of embedding vectors.

        Returns:
            Number of documents stored.
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "The number of documents must match the number of embeddings."
            )

        if not documents:
            return 0

        ids = []
        texts = []
        metadatas = []
        vectors = []

        for index, (document, embedding) in enumerate(
            zip(documents, embeddings)
        ):
            metadata = document["metadata"]

            # Create a deterministic ID for each PDF page chunk.
            document_id = (
                f"{metadata['source']}_"
                f"page_{metadata['page']}_"
                f"chunk_{metadata['chunk']}"
            )

            ids.append(document_id)
            texts.append(document["text"])
            metadatas.append(metadata)
            vectors.append(embedding)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors
        )

        return len(ids)

    def search(self, query_embedding, n_results=5):
        """
        Retrieve the most relevant chunks for a query embedding.
        """

        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        matches = []

        for index, text in enumerate(results["documents"][0]):
            matches.append({
                "text": text,
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index]
            })

        return matches

    def count_documents(self):
        """
        Return the number of stored chunks.
        """

        return self.collection.count()


if __name__ == "__main__":
    store = VectorStoreManager()

    print("===== CHROMADB TEST =====")
    print("Database directory:", store.persist_directory)
    print("Collection:", store.collection.name)
    print("Stored chunks:", store.count_documents())
    print("\n===== CHROMADB INITIALIZATION SUCCESSFUL =====")