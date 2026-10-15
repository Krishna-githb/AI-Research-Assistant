
from src.document_loader import load_pdf

pdf_path = "data/documents/sample_research_paper.pdf"

pages = load_pdf(pdf_path)

target_pages = [1, 6, 12, 14, 16]

for page in pages:
    page_number = page["metadata"]["page"]

    if page_number in target_pages:
        print("\n" + "=" * 70)
        print(f"PAGE {page_number}")
        print("=" * 70)
        print(page["text"])

