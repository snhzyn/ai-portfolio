from core.config import DB, MODEL_DIR
from pipeline.ingest.llm_to_db import LLMtoDatabase

def run_csv_ingestion(file_bytes):
    """
    Run the full CSV ingestion pipeline:
    CSV -> LLM summarization -> classification -> PostgreSQL storage.
    """
    tfidf_vectorizer_path = MODEL_DIR / "vectorizer.pkl"
    svm_model_path = MODEL_DIR / "svm.pkl"
    label_encoder_path = MODEL_DIR / "label.pkl"

    llm_db = LLMtoDatabase(
        host=DB["host"],
        database=DB["database"],
        user=DB["user"],
        password=DB["password"],
        port=DB["port"],
        tfidf_vectorizer_path=tfidf_vectorizer_path,
        svm_model_path=svm_model_path,
        label_encoder_path=label_encoder_path,
    )

    try:
        result = llm_db.csv_to_db(file_bytes)
    finally:
        llm_db.close()

    return result