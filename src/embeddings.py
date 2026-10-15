import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


class EmbeddingManager:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your .env file."
            )

        self.client = genai.Client(api_key=api_key)

        # Google embedding model
        self.model_name = "gemini-embedding-001"

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """

        if not texts:
            return []

        embeddings = []

        for text in texts:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty.")

            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )

            embeddings.append(response.embeddings[0].values)

        return embeddings


if __name__ == "__main__":
    manager = EmbeddingManager()

    sample_texts = [
        "Artificial intelligence helps researchers analyze documents.",
        "Machine learning can support precision agriculture."
    ]

    vectors = manager.generate_embeddings(sample_texts)

    print("===== EMBEDDING TEST =====")
    print("Number of texts:", len(sample_texts))
    print("Number of vectors:", len(vectors))
    print("First vector dimensions:", len(vectors[0]))
    print("First five values:", vectors[0][:5])
    print("\n===== EMBEDDING GENERATION SUCCESSFUL =====")