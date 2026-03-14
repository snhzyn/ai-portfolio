# Content Studio

> **AI Multi-Agent System for Short-Form Video Production**  
> **숏폼 영상 제작을 위한 AI 멀티 에이전트 자동화 시스템**  

> Generate **production-ready short-form video packages** using an AI multi-agent pipeline.  
> 단순한 주제 입력만으로 **스크립트, 스토리보드, 발행 에셋, 영상 생성 프롬프트**를 포함한 제작 준비 완료 패키지를 생성합니다.  

---

## Demo | 데모

**Demo Video** [Youtube](https://www.youtube.com/watch?v=K-E62674dWo)  

**Live Demo** [Link](https://content-studio-v1-910630079560.asia-northeast3.run.app/studio)  
*Note: The live demo is hosted on Google Cloud Run.*  

---

## Project Overview | 프로젝트 소개  

**Content Studio** is an AI-powered production pipeline that automates the early stages of short-form video creation. Instead of manually writing scripts and planning visuals, the system generates a **complete content package** from a single topic input.  

Content Studio는 숏폼 영상 제작의 초기 단계를 자동화하는 AI 기반 생산 파이프라인입니다. 사용자가 주제를 입력하면 시스템은 스크립트 작성, 기획, 관련 미디어 자료 수집을 거쳐 **완성된 콘텐츠 패키지**를 생성합니다.  


The system is designed as a **multi-agent workflow** orchestrated with **LangGraph** and deployed as a cloud service on **GCP**, emphasizing cost-efficiency and scalability.  

Content Studio는 **Google Cloud Platform(GCP)** 인프라 위에서 **LangGraph**를 활용한 멀티 에이전트 오케스트레이션을 구현하여, 복잡한 생성 AI 워크플로우를 안정적인 클라우드 서비스로 배포하는 데 중점을 두었습니다.  

### **Pipeline Process | 처리 과정**  

`User Input → Director → Multi Script Writer → QA Selection → Revision → Storyboard → Title & Thumbnail → Packaging → Print Video Prompt`

**This project demonstrates:**  
- **LLM Multi-agent Orchestration**: Managing complex agent collaboration using LangGraph.  
(LangGraph를 활용한 멀티 에이전트 제어) 
- **AI Content Pipeline**: Designing a pipeline that produces high-quality content without manual prompt engineering.  
(프롬프트 엔지니어링 없이 고품질 콘텐츠를 생산하는 파이프라인 설계)
- **End-to-End Deployment**: Scalable, serverless AI product deployment on Google Cloud.  
(클라우드 기반의 엔드 투 엔드 AI 제품 배포 및 운영)

---

## System Architecture & Tech Stack | 아키텍처 및 기술 스택  

**Cloud Infrastructure (GCP)**
- **Compute: Google Cloud Run**
 - Adopted a serverless architecture capable of **Scale-to-zero** to ensure cost optimization and operational efficiency in response to irregular agent workflow traffic.  
 - 에이전트 워크플로우 특성상 비정기적인 트래픽에 대응하기 위해 Scale-to-zero가 가능한 서버리스 아키텍처를 채택하여 비용 최적화 및 운영 효율성을 확보했습니다. 

- **Container Management: Docker & Google Artifact Registry**
 - Containerized the application to ensure environment consistency and managed security-enhanced images through Artifact Registry.  
 - 애플리케이션을 컨테이너화하여 환경 일관성을 보장하고, Artifact Registry를 통해 보안이 강화된 이미지 관리를 수행합니다.  

- **Region**: `asia-northeast3 (Seoul)` 
 - Deployment in the local region to optimize latency.  
 - 지연 시간 최적화를 위한 국내 리전 배포  

**AI & Software Stack**
- **AI & LLM**: Anthropic Claude (3.5 Sonnet / 3.5 Haiku), LangGraph, LangChain Core
- **Backend**: FastAPI, Pydantic, Python 3.11

---

## System Interface | 시스템 인터페이스 

### Dashboard: Create Section

<p align="center">
  <img src="./assets/images/dashboard.png" alt="Content Studio - Dashboard" width="90%">
</p>

Users provide a Topic, Target Audience, and optional Reference Text. Then choose the Platform, Language, and Duration. Each AI agent performs its role under the coordination of the 'Director' agent.  

사용자는 주제, 목표 청자, 참고 텍스트(옵션)을 입력한 후 게시 플랫폼, 언어, 영상 길이를 선택합니다. '디렉터' 에이전트의 지휘 아래 각 역할을 맡은 AI 에이전트들이 분업을 수행합니다.  

---

### Dashboard: Result Section

<p align="center">
  <img src="./assets/images/.png" alt="Content Studio - Dashboard" width="90%">
</p>

Provides the final package results including generated scripts, scene-by-scene storyboards, and video generation prompts.  

생성된 스크립트, 장면별 스토리보드, 비디오 생성 프롬프트 등 최종 패키지 결과물을 제공합니다.  

---

## Multi-Agent Pipeline | 파이프라인

### **Workflow**

```
User Input(topic, audience, tone, platform)
        ↓
FastAPI API Layer
        ↓
LangGraph Multi-Agent Workflow
(Script Writer → QA → Revision → Storyboard → Assets)
        ↓
Packaging Agent
        ↓
Production Ready Content Package
```

---

### **Key Agents | 주요 에이전트 역할**

* **Script Writer Agents**: Generates multiple script candidates focusing on hooks and narrative structure. (Model: claude-3-5-sonnet)  
* **QA Selection Agent**: Evaluates and selects the strongest script based on engagement potential. (Model: claude-3-5-haiku)  
* **Revision Agent**: Refines pacing, clarity, and Call-to-Action (CTA) of the selected script.  
* **Storyboard Agent**: Transforms the script into a structured visual plan (Visuals, VO, On-screen text).  
* **Title & Thumbnail Agent**: Creates platform-optimized catchy titles and thumbnail text.  
* **Music Agent**: Suggests background music styles aligned with the video's emotional arc.  
* **Packaging Agent**: Produces the final brief and prompts for video generation tools like Runway, Pika, or Sora.  

---  

## Repository Structure | 프로젝트 구조
```text
TBU
```

---

## Environment Setup | 실행 방법

```bash

TBU
```






