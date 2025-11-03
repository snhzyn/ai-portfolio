### **프로젝트 개요**

Aiffel 모두의연구소 데이터사이언티스트 5기 langchain project 입니다.   

모두의연구소 내부 자료를 수집하여, 학생들이 간편히 사용할 수 있는 챗봇 '모두봇'을 생성했습니다.  

Contributors: 추영재, 정소민, 김순호

### **Environment**
- Python 3.12
- langchain 0.3.27
- OpenAI text-embedding-3-large
- OpenAI gpt-4o-mini
- Cohere
- Streamlit
- Poetry

### **Venv**
```Bash
pip install --upgrade pip
pip install poetry
poetry install --no-root    
poetry run jupyter notebook # VSCode: Python Interpreter로 변경
```