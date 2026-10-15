from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager


class ResearchRetriever:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStoreManager()

    def retrieve(self, query, top_k=5):
        """
        Retrieve the most relevant document chunks for a question.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Convert the question into an embedding
        query_embeddings = (
            self.embedding_manager.generate_embeddings([query])
        )

        query_embedding = query_embeddings[0]

        # Search the ChromaDB collection
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k
        )

        return results


if __name__ == "__main__":
    retriever = ResearchRetriever()

    query = "How can AI support precision agriculture?"

    print("===== SEMANTIC SEARCH TEST =====")
    print("Question:", query)

    results = retriever.retrieve(query, top_k=5)

    if not results:
        print("\nNo matching document chunks found.")
    else:
        for index, result in enumerate(results, start=1):
            print(f"\n--- Result {index} ---")
            print("Source:", result["metadata"]["source"])
            print("Page:", result["metadata"]["page"])
            print("Chunk:", result["metadata"]["chunk"])
            print("Distance:", round(result["distance"], 4))
            print("Text:", result["text"][:500])

    print("\n===== RETRIEVAL TEST COMPLETE =====")