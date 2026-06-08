from dotenv import load_dotenv

from src.loader import load_text_file
from src.rag_engine import RAGEngine
from src.splitter import split_text
from src.vector_store import VectorStore


def main() -> None:
    load_dotenv()

    cv_text = load_text_file("data/cv.txt")
    # cv_text = load_text_file("data/cv1.txt")
    job_description = load_text_file("data/job_description.txt")

    print("Loading CV and splitting into chunks...")
    chunks = split_text(cv_text, chunk_size=500, overlap=100)

    print("Building FAISS vector database...")
    vector_store = VectorStore()
    vector_store.build_index(chunks)

    print("Running RAG CV-job match analysis...")
    rag = RAGEngine(vector_store)
    # report = rag.analyse_cv_match(job_description)
    report = rag.analyse_cv_match(job_description)

    print("\n" + "=" * 60)
    print("CV MATCHING REPORT")
    print("=" * 60 + "\n")
    print(report)


if __name__ == "__main__":
    main()
