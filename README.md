### **AI portfolio**  

This repository showcases my work across multiple AI and data science domains, with a strong emphasis on **end-to-end system design, applied machine learning, and real-world data products**.  

In addition to topic-focused projects, this portfolio includes a flagship **end-to-end AI system** developed through collaboration with an industry partner.  

본 포트폴리오는 LangChain 기반 RAG 시스템부터 End-to-End AI 제품까지, 다양한 데이터 사이언스 및 AI 프로젝트를 포함하고 있습니다.  

특히 **추천 시스템, RAG 기반 정보 검색, 시계열 분석, 그리고 End-to-End AI 시스템 설계**에 중점을 두고 있습니다.  

---

### **주요 연구 분야 (Main Areas)**  

- [End-to-End AI Product](https://github.com/snhzyn/ai-portfolio/tree/master/end-to-end-product/NVISIA)  

  [Demo Video](https://www.youtube.com/watch?v=ODp4eys6998) | [Live Demo](https://nvisia-681939980235.asia-northeast3.run.app/)  

  - An AI-driven analytics platform integrating data ingestion, LLM/ML enrichment, PostgreSQL (PostGIS, pgvector), semantic recommendation, and an interactive dashboard.  
  - 북한 관련 뉴스 데이터를 분석하고 LLM 기반 추천 기사 및 시각화를 제공하는 AI 기반 뉴스 인텔리전스 시스템입니다.  

---  

- [RAG & LangChain](https://github.com/snhzyn/ai-portfolio/tree/master/rag-langchain/chatbot)  

  - Multi-format document retrieval   
  - Streamlit-based chatbot applications   
  - PDF, HTML 등 다양한 문서들을 파싱 후 임베딩하여, streamlit을 통해 RAG 챗봇을 구현한 프로젝트입니다.  

---  

- [Recommender Systems](https://github.com/snhzyn/ai-portfolio/tree/master/recommender-system)  

  - AutoInt-based recommendation   
  - Session-based recommendation (MovieLens)   
  - AutoInt 및 세션 기반 추천 시스템을 구현한 프로젝트입니다.  

---  

- [Time-Series Forecasting](https://github.com/snhzyn/ai-portfolio/tree/master/time-series)  

  - DTW-based time-series classification   
  - ARIMA 기반의 시계열 예측 모델 및 DTW 기반의 시계열 데이터 분류 모델입니다.  

---  

- [Algorithm & Problem Solving](https://github.com/snhzyn/ai-portfolio/tree/master/algorithm)
  - Implementation, String, Sorting, Data Structure, Brute Force  
  - Baekjoon 및 Programmers 알고리즘 문제 풀이 아카이브입니다.  

--- 

### **Repo 구조 (Structure)**  

```
AI-PORTFOLIO
├── end-to-end-product
│   └── NVISIA
│       ├── app
│       ├── assets
│       ├── core
│       ├── data
│       ├── pipeline
│       ├── models
│       ├── scripts
│       ├── docker
│       ├── Dockerfile
│       └── main.py
├── rag-langchain
│   └── chatbot
├── recommender-system
│   ├── movielens-autoint
│   └── movielens-session
├── time-series
│   └── usa-electricity-dtw-classification
├── workspace
│   └── algorithm
│       └── baekjoon
└── README.md

```

---

### **기술 스택 (Tech Stack)**

* Languages:  Python, SQL  
* ML / NLP: scikit-learn, TF-IDF, SVC  
* LLM & RAG:  LangChain, OpenAI 
* Data & Storage: PostgreSQL, pgvector, PostGIS  
* Visualization:  Streamlit, Folium, Matplotlib, Plotly  
* Recommender Systems:  AutoInt, Session-based models  
* Time Series:  ARIMA, DTW  
* Infrastructure:  Docker, Google Cloud Run  
* Tools:  Poetry, VSCode, GitHub    

--- 

Last updated / 최종 수정일: **2026-03-11**   
Since July, 2025  

Additional experiments and technical notes that are not included in this portfolio are published on my development blog below.  

포트폴리오에 포함되지 않은 간단한 실험이나 학습 기록은 개인 블로그에 정리하고 있습니다.  

Dev blog  
https://snhzyn.github.io/
