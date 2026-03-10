import pickle
import re
import pandas as pd

class ArticleClassifier:
    """
    Article classification service based on TF-IDF and SVM.
    """

    def __init__(self, tfidf_vectorizer_path, svm_model_path, label_encoder_path):
        self.tfidf_vectorizer = self._load_pickle(tfidf_vectorizer_path)
        self.svm_model = self._load_pickle(svm_model_path)
        self.label_encoder = self._load_pickle(label_encoder_path)

    def _load_pickle(self, path):
        """
        Load TF-IDF vectorizer, SVM classifier, and label encoder from pickle files.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def preprocess_text(self, text):
        """
        Normalize text before TF-IDF transformation.
        """
        if pd.isna(text): 
            return ""
        text = str(text).lower() 
        text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text) 
        return text    

    def predict_category(self, summary, keywords):
        """
        Predict article category using summary and keywords.
        """
        preprocessed_summary = self.preprocess_text(summary)
        preprocessed_keywords = self.preprocess_text(keywords)
        combined_text = preprocessed_summary + " " + preprocessed_keywords

        X_combined = self.tfidf_vectorizer.transform([combined_text])
        svm_pred = self.svm_model.predict(X_combined)[0]
        category = self.label_encoder.inverse_transform([svm_pred])[0]
        return category

