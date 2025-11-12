# Korean Chatbot

MODULABS AIFFEL 간단한 한국어 Q/A 챗봇 프로젝트입니다.  
`ChatbotData.csv`로 Subword 단어장을 만들고 Transformer로 학습하였습니다.   
출처: [songys](https://github.com/songys/Chatbot_data/blob/master/)

## Features
- 한국어 전처리(숫자/문장부호 유지)
- SubwordTextEncoder 
- PAD=0, SOS/EOS
- metric: masked ACC

## Environment
- Python 3.10
- TensorFlow 2.10.0
- Poetry

## Venv
```Bash
pip install --upgrade pip
pip install poetry
poetry install --no-root    
poetry run jupyter notebook # VSCode: Python Interpreter로 변경
```