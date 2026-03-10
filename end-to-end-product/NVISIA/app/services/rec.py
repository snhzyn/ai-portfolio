import psycopg2
from psycopg2.extras import RealDictCursor

"""
Recommender Module

Provides similarity-based article recommendation using pgvector from PostgreSQL.

[Workflow]
1. Retrieve embedding of selected article
2. Compute cosine similarity against other articles
3. Return top-k similar articles
"""

SIMILAR_ARTICLES_QUERY = """
WITH base AS(
    SELECT embedding
    FROM summary
    WHERE id = %s
)
SELECT
    s.id,
    s.keywords,
    s.category,
    s.publish_date,
    s.title,
    s.url,
    1 - (s.embedding <=> b.embedding) AS similarity
FROM summary s
CROSS JOIN base b
WHERE s.id <> %s
ORDER BY s.embedding <=> b.embedding
LIMIT %s;
"""

class Recommender:
    """
    Article recommendation service based on embedding similarity.
    """

    def __init__(self, host, database, user, password, port):

        self.conn = psycopg2.connect(host=host, database=database, user=user, password=password, port=port)
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)

    def get_similar_articles(self, article_id, k=10):
        """
        Return top-k articles similar to the given article_id.        
        """

        try:
            article_id = int(article_id)
        except (TypeError, ValueError):
            raise ValueError(f"유효하지 않은 기사 id입니다: {article_id}")
        
        self.cur.execute(SIMILAR_ARTICLES_QUERY, (article_id, article_id, k))
        rows = self.cur.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "keywords": row["keywords"],
                    "category": row["category"],
                    "publish_date": str(row["publish_date"])[:10],
                    "title": row["title"],
                    "url": row["url"],
                    "similarity": float(row["similarity"])
                }
            )

        return results

    def close(self):
        self.cur.close()
        self.conn.close()