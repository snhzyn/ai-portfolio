import os
import sys
from pathlib import Path
import streamlit.web.cli as stcli

notes = """

How to Run NVISIA | NVISIA 실행 방법

Local Execution (Poetry) | 로컬 실행 (Poetry) :
    1) poetry install
    2) poetry run python main.py

Docker Execution | Docker 실행 :
    docker compose up

※ When running with Docker, PostgreSQL + PostGIS containers will start automatically.
※ Docker 실행 시 PostgreSQL + PostGIS 컨테이너가 자동으로 실행됩니다.

"""

def main():
    dashboard_path = Path(__file__).parent / "app" / "dashboard.py"

    host = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    port = os.getenv("STREAMLIT_SERVER_PORT", "8501")

    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
        f"--server.address={host}",
        f"--server.port={port}",
    ]

    sys.exit(stcli.main())

if __name__ == "__main__":
    print(notes)
    main()