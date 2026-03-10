CREATE TABLE IF NOT EXISTS summary (
    id BIGINT PRIMARY KEY,
    summary TEXT NOT NULL,
    keywords TEXT NOT NULL,
    event_title TEXT,
    event_date TEXT,
    event_person TEXT,
    event_org TEXT,
    event_loc TEXT,
    url TEXT UNIQUE,
    category TEXT,
    title TEXT,
    publish_date DATE,
    embedding VECTOR(3072)
);

CREATE INDEX IF NOT EXISTS idx_summary_embedding
ON summary
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);