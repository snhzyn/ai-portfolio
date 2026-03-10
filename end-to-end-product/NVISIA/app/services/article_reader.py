import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from core.config import DB

class ArticleReader:
    """
    Utility class for reading article data from PostgreSQL.
    """

    def __init__(self, db_config):
        self.db_config = db_config

    def get_psql_conn(self):
        """
        Create a psycopg2 connection for simple queries.
        """
        conn = psycopg2.connect(
            host=self.db_config["host"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            port=self.db_config["port"],
        )
        return conn

    def table_exists(self, conn, table_name, schema):
        """
        Check whether a table exists to prevent errors on first run.
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                );            
                """,
                (schema, table_name),
            )
            return bool(cur.fetchone()[0])

    def load_all_articles(self):
        """
        Load all articles from the summary table into a pandas DataFrame.
        """
        conn = self.get_psql_conn()
        cur = None

        try: 
            if not self.table_exists(conn, "summary", "public"):
                return None

            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    id,
                    title,
                    summary,
                    publish_date,
                    category,
                    event_loc,
                    event_org,
                    url
                FROM summary
                ORDER BY id DESC
            """)

            rows = cur.fetchall()

            df = pd.DataFrame(rows)

            if "publish_date" in df.columns:
                df["publish_date"] = df["publish_date"].astype(str).str[:10]

            return df
        
        finally:
            if cur is not None:
                cur.close()
            conn.close()