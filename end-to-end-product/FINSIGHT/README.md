
Framework: LangGraph
Serving: FastAPI
LLM: gpt-4o-mini
Architecture: multi-agent orchestration backend
MVP endpoint: POST /api/research
Output: structured JSON + markdown report
Deployment: Docker 기반

[structure]
개발: FastAPI > LangGraph > Claude
배포: FastAPI > Docker > Cloud Run
(NVISIA: streamlit > Cloud Run(Docker) >  Cloud SQL

[Agent]
Manager Agent: 사용자 요청 해석, 오늘 briefing scope 결정, worker별 task 생성

SOURCE_REGISTRY = {
    "macro": {
        "primary": ["fed", "ecb", "boj", "bls"],
        "secondary": ["reuters_markets"],
        "max_items": 5
    },

  "worker": "macro",
  "headline": "Fed signals rates may remain higher for longer",
  "publish_date": "2026-03-10",
  "source": "Federal Reserve",
  "source_type": "official",
  "url": "https://...",
  "summary": "The Fed indicated that inflation remains sticky and policy easing may be delayed.",
  "market_impact": "Supports USD and Treasury yields; negative for rate-sensitive equities.",
  "importance": 8,
  "confidence": 0.88,  >>> 설계해야함. 
  "region_tags": ["US", "Global"],
  "asset_tags": ["rates", "usd", "equities"],
  "event_type": "monetary_policy"

confidence =
0.4 * source_reliability +
(공식 기관: 1.0, Reuters: 0.9, 기업 IR: 0.9, CNBC 등 해설 매체: 0.75, 기타 보조 소스: 0.6~0.7)
0.3 * cross_source_confirmation +
(동일 내용이 2개 이상 신뢰 소스에서 확인됨: +가산, 단일 출처만 있음: 낮춤)
0.2 * event_specificity +
(수치/정책 발표/실적처럼 명확한 사실: 높음, 해석/전망/의견 중심: 낮음)
0.1 * timeliness
(사용자가 선택한 날짜와 매우 가까움: 높음, 시차가 큼: 낮춤)

Worker 1. Macro Agent: 금리, CPI, Fed, ECB, BOJ, 경기지표
Federal Reserve FOMC calendars / statements
ECB press releases
BOJ announcements
U.S. BLS release calendar / CPI
필요하면 주요 시장 브리핑용 Reuters Markets

Worker 2. Markets Agent: S&P, Nasdaq, 대형 기술주, 실적, risk sentiment
Reuters Markets
Reuters Global Market Data
필요하면 기업 공식 IR / earnings release 페이지
선택적으로 CNBC 같은 해설성 매체는 보조로만

Worker 3. Commodities & FX Agent: oil, gas, gold, dollar, yields, FX
Reuters Markets → Commodities / Currencies / Rates & Bonds
EIA Weekly Petroleum Status Report
EIA Short-Term Energy Outlook
필요하면 미 재무부 금리 데이터 같은 공식 수치

Worker 4. Geopolitical Risk Agent: 전쟁, 제재, 무역갈등, 공급망, 국가 리스크
Reuters World / Reuters markets-geopolitics coverage
OFAC sanctions
USTR 발표
필요하면 각국 정부/국제기구 공식 발표

Lead Analyst Agent : worker outputs 수집, event normalization, duplicate clustering, 
verification check, country relevance scoring, final ranking, report-ready summary 생성

Report Agent

[버전: 일단은 v1]
v1: Daily Finance Intelligence Report
→ 글로벌/한국 금융 뉴스 수집
→ 섹터별 분석
→ Daily briefing 생성

v2: Korea Equity Impact Layer
→ 오늘 뉴스가 어떤 한국 기업/섹터에 호재·악재인지 연결
→ 상승/하락 가능성 분석

v3: Ranking / signal output
→ “오늘 영향 가능성이 큰 한국 종목 TOP N” 같은 형태





LangGraph Flow

```
START
  ↓
Manager Agent
  ↓
Parallel Workers
  ├─ Macro Agent
  ├─ Markets Agent
  ├─ Commodities & FX Agent
  └─ Geopolitical Risk Agent
  ↓
Lead Analyst Agent
  ↓
Report Agent
  ↓
END
```