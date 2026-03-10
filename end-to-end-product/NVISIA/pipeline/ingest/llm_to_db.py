import os
import pandas as pd
import numpy as np
import re
import io
from openai import OpenAI
from pathlib import Path
import json
import psycopg2
import pickle

from pipeline.ingest.location_normalizer import LocationNormalizer
from pipeline.ingest.article_repository import ArticleRepository
from core.config import OPENAI_MODEL, OPENAI_EMBED_MODEL, NK_CITIES_PATH

"""
LLM ingestion service for NVISIA.

Responsibilities:
- read uploaded CSV files
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

        self._load_models(
            tfidf_vectorizer_path = tfidf_vectorizer_path, 
            svm_model_path = svm_model_path,
            label_encoder_path = label_encoder_path
            )
        
        nk_path = nk_cities_path or NK_CITIES_PATH
        self.location_normalizer = self._init_location_normalizer(nk_path)

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

    def _load_models(self, tfidf_vectorizer_path, svm_model_path, label_encoder_path):
        """
        Load TF-IDF vectorizer, SVM classifier, and label encoder from pickle files.
        """
        with open(tfidf_vectorizer_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        with open(svm_model_path, "rb") as f:
            self.svm_model = pickle.load(f)

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f) 

    def _init_location_normalizer(self, nk_cities_path):
        try:
            return LocationNormalizer(nk_cities_path)
        except Exception as e:
            print(
                f"Warning: Failed to initialize LocationNormalizer. "
                f"Normalization will be skipped. Error: {e}"
            )
            return None   

    # =========================
    # I/O
    # =========================
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

                llm = self.get_article_summary(title, content, publish_date)
                if not llm:
                    stats["failed"] += 1
                    continue

                summary_text = llm.get("summary", "") or ""
                keywords_text = self.value_to_csv_string(llm.get("keywords"))

                category = self.get_category(summary_text, keywords_text)
                embedding = self.get_embeddings(summary_text, keywords_text)

                self.repository.insert_summary(
                    llm=llm,
                    title=title,
                    publish_date=publish_date,
                    url=url,
                    category=category,
                    embedding=embedding,
                    value_to_csv_string = self.value_to_csv_string
                )

                stats["inserted"] += 1     

            except Exception as e:
                stats["failed"] += 1
                print(f"[CSV INGEST ERROR] row={i}, url={row.get('url')}: {e}")

        return stats
    
    # =========================
    # LLM
    # =========================
    def get_article_summary(self, title, contents, publish_date):
        """
        Summarize article and extract structured event fields with LLM.        
        """
        
        prompt = f"""
    아래 기사를 분석하여 요구된 정보를 작성하시오.

    # 기사 제목:
    {title}

    # 기사 내용:
    {contents}

    # 기사 작성일:
    {publish_date}

   1. 아래 형식으로 정리 (괄호안 각 key값의 한글 설명은 참고만 하고 최종 결과에는 포함하지 않음)
    - summary(주요 사건 요약):
    - event_title(사건 주제):
    - event_date(사건 발생일):
    - event_person(사건 핵심 인물):
    - event_org(사건 핵심 조직/기관):
    - event_loc(사건 발생 지명):
    - keywords(주요 키워드):
    
    2. 각 카테고리의 조건
    - "summary": 3 문장 이하로 핵심 내용만 발췌.
    - "event_title": 간단한 한 문장으로 사건 주제 작성.
    - "event_date": yyyy-mm-dd 형식, 기사에 "event_date"가 명시되지 않았으면 "기사 내용" 중 시간 또는 기간을 나타내는 단어(예시로, '어제', '사흘전', '일주일 전' 등)를 참고하여 "기사 작성일" 기준 계산.
    - "event_person": 사건의 주체 인물(들)의 이름만 입력, 다수의 경우 쉼표로 구분.
    - "event_org": 사건의 주체 조직 및 기관의 이름만 입력, 다수의 경우 쉼표로 구분, **언론사명은 반드시 제외**, **신문사명은 반드시 제외**, **기자가 참고한 출처의 이름도 반드시 제외**, **"노동신문"은 반드시 제외**.
    - "event_loc": [도, 시]단위 지명만을 입력하되 "도" 와 "시" 정보가 함께 있는 경우는 반드시 행적구역별로 분리해서 입력. 건물등에서 일어난 사건의 경우는 해당 장소의 [도, 시] 지명을 입력, 행정구역이 "시"일 경우는 꼭 "시"를 명시 (개성시, 평양시, 고성시 등). 
    특히 "평양" / "평양직할시" / "평양시"와 같이 한 지명에 다양한 표기가 있을경우는 "평양시" ([시 이름] + 시)와 같은 형태로 통일. **"북한" 이라는 단어는 반드시 제외**. 북한이 아닌 해외의 사건의 경우만 국가명을 입력.
    - "keywords": "summary", "event_title", "event_person", "event_org", "event_loc" 모두를 종합적으로 고려하여 해당 뉴스 사건을 대표할 수 있는 **단어 5개 선정**, **"북한" 이라는 단어는 반드시 제외**, 쉼표로 구분하여 입력.
    
    - 위 결과를 종합하여 딕셔너리 형태로 출력.
    - 결과를 출력하기 전 다음 체크리스트를 스스로 검증하라:
        - [ ] 내가 사용한 모든 답과 수치는 기사 원문에 존재한다.
        
    - 설명 출력 금지, 답만 출력.
    """

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "당신은 북한 관련 뉴스 사건 정보를 추출하는 전문 분석 모델입니다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0
            )
            
            result_text = response.choices[0].message.content.strip()
            try:
                result = json.loads(result_text)  
            except json.JSONDecodeError:
                print("Parsing error:", result_text)
                return None
            
            if 'event_loc' in result and self.location_normalizer:
                normalized_loc = self.location_normalizer.normalize(result["event_loc"])
                if normalized_loc:
                    result['event_loc'] = normalized_loc
                  
            return result
            
        except Exception as e:
            print("Error in LLM call or parsing:", e)
            return None

    def value_to_csv_string(self, value):
        """
        Normalize list-like or comma-separated values into a single CSV-style string.
        """
        if not value:
            return ""

        if isinstance(value, list):
            return ", ".join(x.strip() for x in value)

        if isinstance(value, str):
            return ", ".join(x.strip() for x in value.split(","))

        return str(value)

    # =========================
    # ML
    # =========================
    def preprocess_text(self, text):

        if pd.isna(text): 
            return ""
        text = str(text).lower() 
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text) 
        return text    

    def get_category(self, summary, keywords):
        """
        Predict article category using TF-IDF and SVM classifier.
        """
        preprocessed_summary = self.preprocess_text(summary)
        preprocessed_keywords = self.preprocess_text(keywords)
        combined_text = preprocessed_summary + " " + preprocessed_keywords

        X_combined = self.tfidf_vectorizer.transform([combined_text])

        svm_pred = self.svm_model.predict(X_combined)[0]

        category = self.label_encoder.inverse_transform([svm_pred])[0]
        return category

    def text_to_embedding(self, text):
        """
        Generate embedding vector for a given text using OpenAI embeddings API.
        """
        text_embeddings = self.client.embeddings.create(
            model = self.embed_model,
            input = text
        )
        embeddings = np.array(text_embeddings.data[0].embedding, dtype = np.float32)
        return embeddings

    def get_embeddings(self, summary, keywords):
        """
        Concatenate summary and keyword embeddings for recommendation use.
        """

        embed_summary = self.text_to_embedding(summary)
        embed_keywords = self.text_to_embedding(keywords)
        embed_rec = np.hstack([embed_summary, embed_keywords])
        return embed_rec
    
    def close(self):
        self.cur.close()
        self.conn.close()