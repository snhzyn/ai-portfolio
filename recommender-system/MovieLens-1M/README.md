### **MovieLens-1M**

UMN(University of Minnesota)의 GroupLens Research 제공 공개 데이터셋입니다.  

[공식 다운로드](https://grouplens.org/datasets/movielens/)  
[Kaggle](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset/data)  

```
Ratings를 통한 영화 추천 Dataset 에 GRU 기반 Session based Recommendation 모델을 설계해보았습니다.   
Reference: Aiffel, GRU4Rec (Hidasi et al., 2015)  
```

### **Environment**
- Python 3.9.7
- TensorFlow-gpu 2.6.0
- Poetry

### **Venv**
```Bash
pip install --upgrade pip
pip install poetry
poetry install --no-root    
poetry run jupyter notebook # VSCode: Python Interpreter로 변경
```