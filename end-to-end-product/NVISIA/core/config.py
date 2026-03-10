import os

"""

Postgres db 설정.

기본값 = localhost, 환경변수 = Docker 실행 시

"""

def get_db_config():
    return dict(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "nvisiaDb"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres1202"),
        port=int(os.getenv("DB_PORT", "5432")),
    )

DB = get_db_config()
