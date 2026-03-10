import os
import pandas as pd
import io
from openai import OpenAI
import psycopg2


from pipeline.ingest.location_normalizer import LocationNormalizer
from pipeline.ingest.article_repository import ArticleRepository
from pipeline.ingest.classifier import ArticleClassifier
from pipeline.ingest.embedding_service import EmbeddingService
from pipeline.ingest.llm_extractor import LLMExtractor
from core.config import OPENAI_MODEL, OPENAI_EMBED_MODEL, NK_CITIES_PATH

"""
LLM ingestion service for NVISIA.

Responsibilities:
- read uploaded CSV files
- coordinate LLM extraction
- classify article categories
- generate embeddings
- store results in PostgreSQL
"""

class LLMtoDatabase:
    """
    Ingestion service that enriches uploaded CSV articles with LLM outputs,
    classifies categories, generates embeddings, and stores results in PostgreSQL.
    """

    def __init__(self, host, database, user, password, port, tfidf_vectorizer_path, svm_model_path, label_encoder_path, nk_cities_path=None):
        """
        Initialize database connection, OpenAI client, ML models,
        and location normalization resources.
        """

        self.llm_model = OPENAI_MODEL
        self.embed_model = OPENAI_EMBED_MODEL

        self._init_openai_client()
        self._init_db(host, database, user, password, port)

        self.repository = ArticleRepository(self.conn, self.cur)

        self.classifier = ArticleClassifier(
            tfidf_vectorizer_path = tfidf_vectorizer_path, 
            svm_model_path = svm_model_path,
            label_encoder_path = label_encoder_path
            )
        
        nk_path = nk_cities_path or NK_CITIES_PATH   
        self.location_normalizer = self._init_location_normalizer(nk_path)

        self.embedding_service = EmbeddingService(
            client = self.client,
            embed_model = self.embed_model
        )
        
        self.llm_extractor = LLMExtractor(
            client=self.client,
            model=self.llm_model,
            location_normalizer=self.location_normalizer,
        )

    def _init_openai_client(self):
        """
        Initialize OpenAI client from environment variable.
        """
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. .env 또는 시스템 환경변수에 설정하세요.")
        
        self.client = OpenAI(api_key=api_key)

    def _init_db(self, host, database, user, password, port):
        """
        Initialize PostgreSQL connection and cursor.
        """
        self.conn = psycopg2.connect(host=host, database=database, user=user, password=password, port=port)
        self.cur = self.conn.cursor()


    def _init_location_normalizer(self, nk_cities_path):
        try:
            return LocationNormalizer(nk_cities_path)
        except Exception as e:
            print(
                f"Warning: Failed to initialize LocationNormalizer. "
                f"Normalization will be skipped. Error: {e}"
            )
            return None   

    @staticmethod
    def read_csv(file_bytes):
        """
        Read uploaded CSV bytes using fallback encodings.
        """
        last_err = None
        for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            except Exception as e:
                last_err = e
        raise last_err
    
    def csv_to_db(self, file_bytes):
        """
        Ingest uploaded CSV bytes into PostgreSQL with enrichment pipeline.        
        """
        df = self.read_csv(file_bytes)

        required_cols = {"title", "content", "publish_date", "url"}
        missing = required_cols - set(df.columns)        
        if missing:
            raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}")       
        
        stats = {
            "total": len(df),
            "inserted": 0,
            "skipped_existing": 0,
            "skipped_empty": 0,
            "failed": 0,
        }

        for i, row in df.iterrows():
            try:
                title = row.get("title", "") or ""
                content = row.get("content", "") or ""
                publish_date = row.get("publish_date", "") or ""
                url = row["url"]

                if not isinstance(content, str) or not content.strip():
                    stats["skipped_empty"] += 1
                    continue

                if self.repository.exists_by_url(url):
                    stats["skipped_existing"] += 1
                    continue

                llm = self.llm_extractor.extract_article_summary(title, content, publish_date)
                if not llm:
                    stats["failed"] += 1
                    continue

                summary_text = llm.get("summary", "") or ""
                keywords_text = self.llm_extractor.value_to_csv_string(llm.get("keywords"))

                category = self.classifier.predict_category(summary_text, keywords_text)
                embedding = self.embedding_service.get_embeddings(summary_text, keywords_text)

                self.repository.insert_summary(
                    llm=llm,
                    title=title,
                    publish_date=publish_date,
                    url=url,
                    category=category,
                    embedding=embedding,
                    value_to_csv_string = self.llm_extractor.value_to_csv_string
                )

                stats["inserted"] += 1     

            except Exception as e:
                stats["failed"] += 1
                print(f"[CSV INGEST ERROR] row={i}, url={row.get('url')}: {e}")

        return stats
       
    def close(self):
        self.cur.close()
        self.conn.close()