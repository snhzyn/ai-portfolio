import json
from openai import OpenAI

class LLMExtractor:
    """
    LLM-based article enrichment service for summary and structured event extraction.
    """

    def __init__(self, client, model, location_normalizer=None):
        self.client = client
        self.model = model
        self.location_normalizer = location_normalizer

    def extract_article_summary(self, title, contents, publish_date):
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
                model=self.model,
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
            
            if "event_loc" in result and self.location_normalizer:
                normalized_loc = self.location_normalizer.normalize(result["event_loc"])
                if normalized_loc:
                    result['event_loc'] = normalized_loc
                  
            return result
            
        except Exception as e:
            print("Error in LLM call or parsing:", e)
            return None

    @staticmethod
    def value_to_csv_string(value):
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