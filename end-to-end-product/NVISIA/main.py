import sys
from pathlib import Path
import streamlit.web.cli as stcli

notes = """

서비스를 이용하기 위해서는 PostgreSQL 및 이의 확장 기능인 pgvector와 postgis가 사전에 설치되어 있어야 합니다.

데이터베이스 정보(host, user, password)는 사용 중인 환경에 맞게 수정이 필요합니다.

실행 방법:
    1) poetry install
    2) poetry run python main.py

"""

def main():
    dashboard_path = Path(__file__).parent / "src" / "dashboard.py"
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
