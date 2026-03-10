import argparse
from pipeline.crawling.spn_crawler import SPNCrawler

"""

크롤러 실행 스크립트입니다. 아래는 사용 예시입니다. 

py scripts/run_spn_crawler.py --start 101404 --end 100000 --filename data/raw/spnews_db_test.csv --section 북한N

"""

def main():
    parser = argparse.ArgumentParser(
        description = "SPN 뉴스 사이트 기사 크롤러"
    )

    parser.add_argument(
        "--start",
        type = int,
        required = True,
        help = "시작 idx 번호 (예: 101404)"
    )

    parser.add_argument(
        "--end",
        type = int,
        required = True,
        help = "끝 idx 번호 (예: 100000)"
    )

    parser.add_argument(
        "--filename",
        type = str,
        required = True,
        help = "저장할 CSV 파일명"
    )

    parser.add_argument(
        "--section",
        type = str,
        default = None,
        help = "섹션 필터 (예: 북한N)"
    )

    parser.add_argument(
        "--sleep-min",
        type = float,
        default = 0.5,
        help = "요청 최소 대기 시간"
    )

    parser.add_argument(
        "--sleep-max",
        type = float,
        default = 1.0,
        help = "요청 최대 대기 시간"
    )

    args = parser.parse_args()

    crawler = SPNCrawler(
        start = args.start,
        end = args.end,
        filename = args.filename,
        section = args.section,
        sleep_min = args.sleep_min,
        sleep_max = args.sleep_max
    )

    crawler.run()


if __name__ == "__main__":
    main()