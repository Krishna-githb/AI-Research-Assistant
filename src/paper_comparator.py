import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

from src.document_loader import load_pdf


class PaperComparator:
    """Compare two research papers using evidence from their full text."""

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Check that your .env file exists in the project root."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.8-flash"

    @staticmethod
    def _extract_paper_text(pdf_path):
        """Extract readable text from a PDF and preserve page numbers."""

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {path.resolve()}"
            )

        pages = load_pdf(path)

        page_texts = []

        for page in pages:
            page_number = page["metadata"]["page"]
            text = page["text"]

            page_texts.append(
                f"\n--- Page {page_number} ---\n{text}"
            )

        return "\n".join(page_texts)

    def compare_papers(self, pdf_path_1, pdf_path_2):
        """Generate an evidence-grounded comparison of two research papers."""

        paper_1 = self._extract_paper_text(pdf_path_1)
        paper_2 = self._extract_paper_text(pdf_path_2)

        print("Both PDFs were loaded successfully.")
        print(f"Paper 1 extracted characters: {len(paper_1):,}")
        print(f"Paper 2 extracted characters: {len(paper_2):,}")
        print("Generating the comparison report...")

        prompt = f"""
You are an AI research assistant specializing in scientific literature
analysis.

Compare the following two research papers using ONLY the evidence
provided in their extracted text.

PAPER 1
Source filename: {Path(pdf_path_1).name}

{paper_1}


PAPER 2
Source filename: {Path(pdf_path_2).name}

{paper_2}


INSTRUCTIONS

1. Do not invent titles, authors, publication details, methods, datasets,
   results, numerical values, or citations.

2. If information is unavailable or unclear, write:
   "Not explicitly stated in the provided paper text."

3. Distinguish clearly between:
   - Methods that were proposed.
   - Methods that were implemented.
   - Methods that were experimentally evaluated.
   - Results that were actually reported.

4. Do not describe a conceptual framework or decision model as a
   machine-learning experiment unless the paper supports that claim.

5. Preserve important technical terminology used by the authors.

6. For important findings, include the relevant page number when it can
   be determined from the provided page markers.

7. Do not claim that a research gap exists merely because a topic is not
   mentioned in one paper. Identify potential gaps as interpretations
   and explain the evidence supporting them.

8. If the extracted text is insufficient to support a conclusion, state
   that limitation explicitly.

Generate a structured Markdown report using the following sections.

# Comparative Research Paper Analysis

## 1. Paper Information
For each paper, identify the title, authors, publication venue, year,
and research domain where the information is available.

## 2. Research Problem and Objectives
Explain the problem each paper addresses and its stated objectives.

## 3. Methodology
Describe the research approach used in each paper, including whether
it is experimental, empirical, conceptual, review-based, or another
approach, but only when supported by the text.

## 4. AI/ML or Decision-Modelling Techniques
Compare the techniques, models, frameworks, algorithms, and decision
support approaches discussed or evaluated in each paper.

## 5. Datasets and Experimental Setup
Identify datasets, study locations, sample sizes, experimental
conditions, evaluation metrics, and validation methods where reported.
Clearly distinguish actual experiments from proposed implementations.

## 6. Key Findings and Results
Summarize the major findings from each paper. Include numerical results
only when explicitly supported by the text.

## 7. Limitations
Compare the limitations acknowledged by the authors. Separate
author-reported limitations from limitations inferred from the text.

## 8. Similarities and Differences
Present a comparison table with these columns:
- Comparison dimension
- Paper 1
- Paper 2
- Main difference or similarity

## 9. Research Gaps and Future Opportunities
Identify potential opportunities for further research based on the
comparison. Label these as interpretations rather than established
facts when the papers do not explicitly identify them.

## 10. Overall Synthesis
Explain what each paper contributes, how their approaches relate,
and what a researcher can reasonably learn from reading both papers.

## 11. Evidence and Uncertainty
List important conclusions that could not be verified from the extracted
text and any extraction limitations that may affect the comparison.
"""

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty comparison report."
                    )

                return response.text

            except ServerError as error:
                if attempt == 2:
                    raise

                wait_seconds = 5 * (attempt + 1)

                print(
                    f"Gemini server error. Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

        raise RuntimeError("Could not generate the comparison report.")

    @staticmethod
    def save_report(report, output_path):
        """Save the comparison report as a Markdown file."""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report, encoding="utf-8")

        print(f"\nComparison report saved to: {output_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two research papers using Gemini."
    )

    parser.add_argument(
        "paper1",
        help="Path to the first research paper PDF.",
    )

    parser.add_argument(
        "paper2",
        help="Path to the second research paper PDF.",
    )

    parser.add_argument(
        "--output",
        default="comparison_report.md",
        help="Output Markdown file path (default: comparison_report.md).",
    )

    args = parser.parse_args()

    try:
        comparator = PaperComparator()

        report = comparator.compare_papers(
            args.paper1,
            args.paper2,
        )

        comparator.save_report(report, args.output)

        print("\nComparison completed successfully.")

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nError: {error}")

    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()

