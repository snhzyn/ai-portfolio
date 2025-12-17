# PROJECT NVISIA  
## North Korea Visual Insight with SIA

> **End-to-End AI Product for North Korea News Intelligence**  
> Developed as part of **ModuLabs Aiffel Data Scientist Bootcamp**  
> In collaboration with **SI Analytics** (https://si-analytics.ai/)

---

### 프로젝트 개요

Aiffel 모두의연구소 데이터사이언티스트 5기 **End-to-End Product 프로젝트**입니다.  

본 프로젝트는 북한 관련 뉴스를 자동으로 수집·분석·시각화하여,  
연구자·정책 분석가·기업 실무자가 북한 동향을 **직관적으로 탐색**할 수 있도록 지원하는  
AI 기반 분석 시스템을 구축하는 것을 목표로 합니다.  

NVISIA는 단순 모델 구현이 아닌,

> **데이터 수집 → ETL → DB 설계 → ML/LLM → 추천 시스템 → 대시보드**

까지 전 과정을 포함하는 프로젝트입니다.

팀의 개발 과정, 실험 로그, 이슈 관리 내역은 아래 GitHub 저장소에 정리되어 있습니다.
> https://github.com/milkpotato1000/project_NVISIA

---

### 프로젝트 목표

- 북한 관련 뉴스 자동 수집 및 데이터베이스 저장 
- PostgreSQL 기반 데이터베이스 구축 및 관리      
- LLM 기반 사건 요약, 키워드 추출  
- ML 기반 뉴스 카테고리 분류  
- 지리·시간·이슈 관점의 분석을 위한 데이터 모델링(postgis 활용)  
- 추천 시스템 및 Knowledge-graph 기반 연관 뉴스 탐색  
- Streamlit 기반 **인터랙티브 대시보드** 제공  

---

### 프로젝트 기간

- **2025.11.17 ~ 2026.01.06** *(진행 중)*  
- 현재 상태(2025-12-17): 핵심 데이터 파이프라인 및 MVP 기능 구현 완료

---

### Contributors

- 손호진
- 천승우
- 정소민  
- 진용현  

---

### Repository Structure

```
project-NVISIA/
 ├─ src/              # python source files
 ├─ models/           # SVM, vectorizer, labeling pickle files
 ├─ data/             # csv files, experiments, notes
 ├─ crawling/         # news crawler
 ├─ main.py           # starting python file
 ├─ pyproject.toml    # poetry requirements
 ├─ .gitignore
 └─ README.md
```

---

### System Architecture

```
User-provided CSV (news articles)
(test input: News Crawling via BeautifulSoup)
        ↓
Data cleaning & ETL
        ↓
LLM / ML enrichment
 ├─ Summary & Keywords extraction (model: gpt-4o-mini)
 ├─ Category classification (SVC)
 └─ Text embedding generation (model: text-embedding-ada-002)
        ↓
Postgresql data store
 ├─ Enriched articles
 │   ├─ summary
 │   ├─ keywords
 │   ├─ category
 │   ├─ event date / person / org / loc 
 │   ├─ title
 │   ├─ url, publish_date
 │   └─ embedding (pgvector)
 ├─ nk_org (North Korea essential organization geodata, EPSG:5179)
 └─ geojson (geospatial data, EPSG:4326)
        ↓
Analytics & Retrieval Layer (Query-time)
 ├─ Similarity-based recommendation
 ├─ Knowledge graph construction (TF-IDF / NetworkX)
 └─ Spatial queries: join nk_org & geojson (postgis)
        ↓
Streamlit Dashboard
 ├─ News table & filters
 ├─ Charts
 ├─ Interactive map (folium)
 ├─ Recommended articles
 └─ Knowledge graph visualization
```

---

### Key Features

#### 1. Data Pipeline
- BeautifulSoup 기반 북한 뉴스 크롤링 (본 project에서는 spnews)
- CSV 입력 데이터 정제 및 ETL 수행
- 다중 언론사 지원
- URL 기준 중복 기사 제거

#### 2. LLM 기반 정보 추출
- 기사 핵심 내용 요약 (3문장 이내)
- 사건 발생일 추론 (publish_date를 기준으로 상대적 시간 표현을 고려)
- 주요 인물, 기관, 지명, 키워드 추출
- 반복적인 prompt engineering을 통해 LLM 추출 정확도 개선
- 비용 분석을 바탕으로 경량 LLM(gpt-4o-mini) 활용

#### 3. Database & Storage
- Postgresql 기반 정규화된 뉴스 및 LLM 출력 데이터 저장(LLM >> PostgresDB)
- pgvector를 이용한 임베딩 저장 및 유사도 검색
- postgis를 활용한 북한 행정구역 공간 데이터 관리

#### 4. Recommendation System
- Summary + Keywords 임베딩 기반 콘텐츠 추천
- cosine similarity를 통해 콘텐츠 기반 추천
- pgvector sql 연산으로 DB 내 빠른 계산
- 임베딩 모델 성능 비교 실험(text-embedding-small vs text-embedding-ada-002)
- 추천 결과 정성적 평가 진행

#### 5. Geospatial Analytics
- 북한 행정구역(도/시/광역시) 정규화 규칙 설계(ex. 함북 > 함경북도)
- Nominatim(OpenStreetMap) 기반 좌표 변환 (EPSG:5179 > EPSG:4326)
- Folium 기반 지도 시각화

#### 6. Streamlit Dashboard
- 뉴스 리스트 및 필터링
- 카테고리 시각화
- 지도 기반 기사 내 지리 정보 시각화
- 선택 기사 관련 기사들 추천
- 추천 기사들의 keywords 기반 Knowledge graph 시각화

---

### Tech Stack

#### Environment
- Python **3.12**
- Poetry

#### Core Libraries
- OpenAI API
- psycopg2
- pandas
- scikit-learn
- Streamlit
- folium 
- BeautifulSoup (bs4)
- PostgresSQL
- pgvector 
- postgis

---

### Environment Setup

```bash
pip install --upgrade pip
pip install poetry
poetry install --no-root
poetry run python main.py
```

---

### Work in Progress

- 분류·추천 모델 정량 평가 지표 추가
- 대용량 데이터 대응 (pagination, query optimization)
- 반경 기반 공간 쿼리(postgis buffer 활용)
- 검색 기능 고도화
