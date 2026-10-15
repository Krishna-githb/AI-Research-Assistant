import streamlit as st
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from src.summarizer import ResearchSummarizer
from src.document_loader import load_pdf


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()


@st.cache_resource
def get_summarizer():
    return ResearchSummarizer()


st.title("📚 AI Research Assistant")

st.markdown(
    """
    Your AI-powered research companion for exploring,
    understanding, and summarizing academic papers.
    """
)

st.divider()

qa_tab, summary_tab = st.tabs(
    ["💬 Research Q&A", "📄 Paper Summarizer"]
)


# --------------------------------------------------
# TAB 1: QUESTION ANSWERING
# --------------------------------------------------

with qa_tab:
    st.header("Ask Questions About Research")

    question = st.text_area(
        "Enter your research question",
        placeholder=(
            "How can AI-based decision support systems "
            "improve precision agriculture?"
        ),
        height=120,
        key="research_question"
    )

    if st.button("Ask Research Assistant", type="primary"):
        if not question.strip():
            st.warning("Please enter a research question.")

        else:
            try:
                with st.spinner(
                    "Searching papers and generating an answer..."
                ):
                    rag = get_rag_pipeline()
                    result = rag.answer_question(
                        question,
                        top_k=5
                    )

                st.subheader("Answer")
                st.markdown(result["answer"])

                st.subheader("Sources")

                if result["sources"]:
                    for source in result["sources"]:
                        with st.expander(
                            f"{source['source']} — "
                            f"Page {source['page']}, "
                            f"Chunk {source['chunk']}"
                        ):
                            st.write("PDF:", source["source"])
                            st.write("Page:", source["page"])
                            st.write("Chunk:", source["chunk"])
                            st.write(
                                "Retrieval distance:",
                                round(source["distance"], 4)
                            )
                else:
                    st.info("No source passages were retrieved.")

            except Exception as error:
                st.error(
                    "The assistant could not complete your request. "
                    "Check the terminal for details."
                )
                print(f"RAG error: {error}")


# --------------------------------------------------
# TAB 2: RESEARCH PAPER SUMMARIZATION
# --------------------------------------------------

with summary_tab:
    st.header("Summarize a Research Paper")

    st.write(
        "Upload a PDF to generate a structured academic summary."
    )

    uploaded_pdf = st.file_uploader(
        "Choose a research paper",
        type=["pdf"],
        key="summary_pdf"
    )

    if uploaded_pdf is not None:
        st.success(f"Selected: {uploaded_pdf.name}")

        if st.button("Generate Summary", type="primary"):
            try:
                with st.spinner(
                    "Extracting the paper and generating its summary..."
                ):
                    # Save the uploaded PDF temporarily.
                    temporary_directory = Path("data/uploads")
                    temporary_directory.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    temporary_path = (
                        temporary_directory / uploaded_pdf.name
                    )

                    temporary_path.write_bytes(
                        uploaded_pdf.getvalue()
                    )

                    # Validate that text can be extracted.
                    pages = load_pdf(temporary_path)

                    if not pages:
                        raise ValueError(
                            "No readable text was found in the PDF."
                        )

                    summarizer = get_summarizer()

                    summary = summarizer.summarize_pdf(
                        str(temporary_path)
                    )

                st.subheader("Research Paper Summary")
                st.markdown(summary)

                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="research_paper_summary.md",
                    mime="text/markdown"
                )

            except Exception as error:
                st.error(
                    "Unable to summarize this PDF. "
                    "Check the terminal for details."
                )
                print(f"Summarization error: {error}")


st.divider()

st.caption(
    "AI Research Assistant | Gemini, ChromaDB, LangChain, Streamlit"
)