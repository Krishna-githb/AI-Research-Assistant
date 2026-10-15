from src.retriever import ResearchRetriever
from google import genai
from dotenv import load_dotenv
import os


load_dotenv()


class RAGPipeline:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.8-flash"
        self.retriever = ResearchRetriever()

    def answer_question(self, question, top_k=5):
        """
        Generate an answer using retrieved research-paper passages.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1: Retrieve relevant document chunks
        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=top_k
        )

        if not retrieved_chunks:
            return {
                "answer": (
                    "No relevant research passages were found. "
                    "Please ingest research papers before asking questions."
                ),
                "sources": []
            }

        # Step 2: Build the evidence context
        context_parts = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk["metadata"]

            source_info = (
                f"Source: {metadata['source']}, "
                f"Page: {metadata['page']}, "
                f"Chunk: {metadata['chunk']}"
            )

            context_parts.append(
                f"[Evidence {index}]\n"
                f"{source_info}\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        # Step 3: Construct the prompt
        prompt = f"""
You are an AI research assistant.

Answer the user's question using only the evidence provided below.

Instructions:
1. Base factual claims on the supplied research passages.
2. Do not invent facts, statistics, citations, or findings.
3. If the evidence does not contain enough information, clearly say so.
4. Explain the answer clearly and organize it into readable paragraphs.
5. Cite supporting passages using their evidence labels, such as [Evidence 1].
6. Distinguish the paper's claims from any conclusions that cannot be established
   from the supplied passages.

RESEARCH EVIDENCE:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        # Step 4: Generate the answer
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        answer = response.text or (
            "Gemini returned no text for this question."
        )

        # Step 5: Return the answer and its source metadata
        sources = []

        for chunk in retrieved_chunks:
            sources.append({
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "chunk": chunk["metadata"]["chunk"],
                "distance": chunk["distance"]
            })

        return {
            "answer": answer,
            "sources": sources
        }


if __name__ == "__main__":
    rag = RAGPipeline()

    question = (
        "How can AI-based decision support systems "
        "improve precision agriculture?"
    )

    print("=" * 60)
    print("AI RESEARCH ASSISTANT - RAG TEST")
    print("=" * 60)

    print("\nQuestion:")
    print(question)

    result = rag.answer_question(question, top_k=5)

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:
        print(
            f"- {source['source']} | "
            f"Page {source['page']} | "
            f"Chunk {source['chunk']}"
        )

    print("\n===== RAG PIPELINE TEST COMPLETE =====")