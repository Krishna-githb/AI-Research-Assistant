import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

from src.document_loader import load_pdf


load_dotenv()


class ResearchSummarizer:
    def __init__(self):
        """Initialize the Gemini client."""

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your .env file."
            )

        self.client = genai.Client(api_key=api_key)

        # Use the model configured for your project.
        self.model_name = "gemini-3.8-flash"

    def summarize_pdf(self, pdf_path):
        """Extract a PDF and generate a structured research summary."""

        # Step 1: Extract text from the PDF.
        pages = load_pdf(pdf_path)

        paper_text = "\n\n".join(
            f"Page {page['metadata']['page']}:\n{page['text']}"
            for page in pages
        )

        if not paper_text.strip():
            raise ValueError(
                "The PDF does not contain extractable text."
            )

        print(f"Extracted {len(pages)} pages.")

        # Step 2: Create the summarization prompt.
        prompt = f"""
You are an academic research assistant.

Analyze the research paper provided below and produce a
structured summary.

Use only information supported by the paper.

RULES:
1. Do not invent facts, statistics, datasets, experiments, or results.
2. Distinguish proposed methods from methods actually evaluated.
3. Report numerical results only when explicitly stated in the paper.
4. Do not invent author-stated limitations or future research directions.
5. If information is missing, write "Not explicitly stated in the paper."
6. Preserve the paper's technical terminology.

Use this structure:

# Research Paper Summary

## 1. Paper Information
- Title
- Authors
- Publication or journal information
- Research domain

## 2. Research Problem

## 3. Research Objectives

## 4. Proposed Methodology

## 5. AI/ML Techniques
Distinguish proposed techniques from those actually evaluated.

## 6. Key Findings
Include numerical results only when supported by the paper.

## 7. Limitations
Report limitations explicitly discussed by the authors.

## 8. Future Research Directions

## 9. Overall Takeaway

RESEARCH PAPER TEXT:
{paper_text}
"""

        # Step 3: Generate the summary with retries for HTTP 503 errors.
        response = None

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                break

            except ServerError as error:
                if error.code != 503 or attempt == 2:
                    raise

                wait_seconds = 5 * (2 ** attempt)

                print(
                    f"Gemini is temporarily unavailable. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

        # Step 4: Validate and return the generated summary.
        summary = response.text if response else None

        if not summary or not summary.strip():
            raise RuntimeError(
                "Gemini did not return a summary."
            )

        return summary


if __name__ == "__main__":
    summarizer = ResearchSummarizer()

    pdf_path = "data/documents/sample_research_paper.pdf"

    print("=" * 60)
    print("AI RESEARCH ASSISTANT - PAPER SUMMARIZATION")
    print("=" * 60)

    print(f"\nPDF: {pdf_path}")
    print("\nGenerating summary. Please wait...\n")

    summary = summarizer.summarize_pdf(pdf_path)

    print("\n" + "=" * 60)
    print("RESEARCH PAPER SUMMARY")
    print("=" * 60)

    print(summary)

    print("\n" + "=" * 60)
    print("SUMMARIZATION TEST SUCCESSFUL")
    print("=" * 60)