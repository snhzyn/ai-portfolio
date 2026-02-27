### **AI portfolio**  

본 포트폴리오는 LangChain과 LLM을 활용한 RAG 시스템부터 End-to-End AI 프로젝트까지, 
다양한 데이터 사이언스 프로젝트를 포함하고 있습니다.  

아래의 주요 분야에서 확인하실 수 있듯이, RAG, 추천시스템, 시계열 분석에 특화하였습니다.   

This repository showcases my work across multiple AI and data science domains,  
with a strong emphasis on **end-to-end system design, applied machine learning, and real-world data products**.

In addition to topic-focused projects, this portfolio includes a flagship **end-to-end AI product** developed in collaboration with an industry partner.  

---

### **주요 연구 분야 (Main Areas)**  

- [End-to-End AI Product](https://github.com/snhzyn/ai-portfolio/tree/master/end-to-end-product/project-NVISIA)
  - An AI-driven analytics system integrating data ingestion, LLM/ML enrichment, PostgreSQL (PostGIS, pgvector), query-time recommendation, and an interactive dashboard.    
  - 북한 관련 뉴스 데이터를 분석하여 LLM 기반 추천 기사를 대시보드 형태로 제공하는 AI 분석 시스템입니다.   

- [RAG & LangChain](https://github.com/snhzyn/ai-portfolio/tree/master/rag-langchain/chatbot)  
  - Multi-format document retrieval   
  - Streamlit-based chatbot applications   
  - PDF, HTML 등 다양한 문서들을 파싱 후 임베딩하여, streamlit을 통해 RAG 챗봇을 구현한 프로젝트입니다.  

- [Recommender Systems](https://github.com/snhzyn/ai-portfolio/tree/master/recommender-system)  
  - AutoInt-based recommendation   
  - Session-based recommendation (MovieLens)   
  - AutoInt 및 세션 기반 추천 시스템입니다.  

- [Time-Series Forecasting](https://github.com/snhzyn/ai-portfolio/tree/master/time-series)
  - ARIMA-based forecasting   
  - DTW-based time-series classification   
  - ARIMA 기반의 시계열 예측 모델과 DTW 기반의 시계열 데이터 분류 모델입니다.  

- [Algorithm & Problem Solving](https://github.com/snhzyn/ai-portfolio/tree/master/algorithm)
  - Implementation, String, Sorting, Data Structure, Brute Force
  - Baekjoon, programmers 등 알고리즘 문제 풀이 아카이브입니다.

--- 

### **Repo 구조 (Structure)**  

```
AI-PORTFOLIO
├── end-to-end-product
│   └── project-NVISIA
│       ├── crawling
│       ├── data
│       ├── models
│       ├── src
│       └── main.py
├── rag-langchain
│   └── chatbot
├── recommender-system
│   ├── movielens-autoint
│   └── movielens-session
├── time-series
│   ├── airpassengers-prediction
│   └── usa-electricity-dtw-classification
├── workspace
│   ├── algorithm
│   │   └── baekjoon
│   ├── deep-learning
│   │   ├── computer-vision
│   │   ├── nlp
│   │   └── prediction
│   └── machine-learning
│       ├── classification
│       ├── eda
│       ├── nlp
│       └── prediction
└── README.md

```

---

### **기술 스택 (Tech Stack)**

* Languages: Python, SQL
* ML / NLP: scikit-learn, TF-IDF, SVC
* LLM & RAG: LangChain, OpenAI
* Data & Storage: PostgreSQL, pgvector, postgis
* Visualization: Streamlit, Folium, Matplotlib, Plotly
* Recommender Systems: AutoInt, Session-based models
* Time Series: ARIMA, DTW
* Tools: Poetry, VSCode, GitHub, Docker

---

### **기타 (Workspace)**

위의 프로젝트 외에 진행한 단순 실험 및 노트북은 `/workspace` 에 아카이브 형태로 저장해두었습니다.  
머신 러닝 모델 및 딥러닝 모델을 기반으로한 다양한 실험들을 확인하실 수 있습니다. 

Additional studies, experiments, and archived projects are stored under the `/workspace` directory.  
Includes early-stage models, EDA(Exploratory Data Analysis), and foundational deep-learning exercises.

[Workspace](https://github.com/snhzyn/ai-portfolio/tree/master/workspace)


Last updated / 최종 수정일: **2025-12-19**   
Since July, 2025  

--- 

Portfolio 에 담겨있지 않은 단순한 작업 및 연구는 개인 blog에 업로드하고 있으며,  
아래 링크를 통해 확인하실 수 있습니다.   

Additional work not included in this portfolio is posted on my blog at the link below.  

[Dev blog](https://snhzyn.github.io/)
