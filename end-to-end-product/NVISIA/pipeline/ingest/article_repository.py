class ArticleRepository:
    """
    Repository layer for summary table access.
    Handles duplicate checks and article insertion.
    """

    def __init__(self, conn, cur):
        self.conn = conn
        self.cur = cur

    def exists_by_url(self, url):
        """
        Return True if the article URL already exists in summary table.
        """

        query = """
            SELECT COUNT(*) 
            FROM summary
            WHERE url = %s;
        """

        self.cur.execute(query, (url,))
        count = self.cur.fetchone()[0]
        return count > 0

    def insert_summary(self, llm, title, publish_date, url, category, embedding, value_to_csv_string):
        """
        Insert enriched article record into summary table.
        """

        query = """
            INSERT INTO summary
                (summary, keywords, event_title, event_date,
                 event_person, event_org, event_loc, url, title, publish_date, category, embedding)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING;
        """

        values = (
            llm.get("summary"),
            value_to_csv_string(llm.get("keywords")),
            llm.get("event_title"),
            llm.get("event_date"),
            value_to_csv_string(llm.get("event_person")),
            value_to_csv_string(llm.get("event_org")),
            value_to_csv_string(llm.get("event_loc")),
            url,
            title, 
            publish_date,
            category,
            embedding.tolist(),
        )

        try:
            self.cur.execute(query, values)
            self.conn.commit()

            if self.cur.rowcount == 0:
                print(f"[DB INSERT ERROR] 이미 존재하는 기사입니다. url={url}")

        except Exception as e:
            self.conn.rollback()
            print(f"[DB INSERT ERROR] url={url} ⇒ {e}")