import numpy as np
from openai import OpenAI

class EmbeddingService:
    """
    Embedding service for article recommendation features.
    """

    def __init__(self, client, embed_model):
        self.client = client
        self.embed_model = embed_model 

    def text_to_embedding(self, text):
        """
        Generate embedding vector for a given text using OpenAI embeddings API.
        """
        text_embeddings = self.client.embeddings.create(
            model = self.embed_model,
            input = text
        )
        embeddings = np.array(text_embeddings.data[0].embedding, dtype = np.float32)
        return embeddings

    def get_embeddings(self, summary, keywords):
        """
        Concatenate summary and keyword embeddings for recommendation use.
        """
        embed_summary = self.text_to_embedding(summary)
        embed_keywords = self.text_to_embedding(keywords)
        embed_rec = np.hstack([embed_summary, embed_keywords])
        return embed_rec