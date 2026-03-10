import requests
from bs4 import BeautifulSoup 
import csv
from datetime import datetime

import os
import time
import random

"""

www.spnews.co.kr
북한 전문 인터넷 기사 크롤러입니다.

본 크롤러는 main app과 연동되지 않습니다.
scripts 폴더의 run_spn_cralwer.py 를 통해 app 구동에 필요한 기사 데이터를 다운받을 수 있습니다.

"""

BASE_URL = "https://www.spnews.co.kr/news/articleView.html?idxno={}"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

class ArticleCrawler:

    def __init__(self, url):
        self.url = url
        self.soup = self._get_soup()

    def _get_soup(self):
        response = requests.get(self.url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    
    def get_article_url(self):
        meta_tag = self.soup.find("meta", property = "og:url")
        return meta_tag["content"] if meta_tag else None
    
    def get_id(self):
        url = self.get_article_url()    
        if not url:
            return None

        # source(spnew, ytn 등등)
        source = url.split("//")[1]
        source = source.split("/")[0]
        source = source.replace("www.", "").split(".")[0].lower()

        # id
        if "idxno" not in url:
            return None
        
        idx = url.split("idxno=")[1]
        return f"{source}_{idx}"
    
    def get_source(self):
        url = self.get_article_url()
        if not url:
            return None

        source = url.split("//")[1]
        source = source.split("/")[0]
        source = source.replace("www.", "").split(".")[0].lower()
        return source
    
    def get_title(self):
        header = self.soup.find("h1", class_ = "heading")
        return header.get_text(strip = True) if header else None
      
    def get_author(self):
        tag = self.soup.find("i", class_ = "icon-user-o")
        if tag and tag.parent:
            author = tag.parent.get_text(strip=True)
            return author.replace("기자명", "").strip()
        return None
        
    def get_section(self):
        breadcrumb = self.soup.find("ul", class_="breadcrumbs")
        if breadcrumb:
            for a in breadcrumb.find_all("a"):
                text = a.get_text(strip = True)
                if text in ["북한N", "서울&", "한반도W", "이슈+"]:   
                    return text
        return None
              
    def get_category(self):
        breadcrumb = self.soup.find("ul", class_ = "breadcrumbs")
        if breadcrumb:
            for a in breadcrumb.find_all("a"):
                text = a.get_text(strip = True)
                if text in ["정치", "외교", "군사", "경제/산업", "사회/문화/체육"]:
                    return text
    
    def get_publish_date(self):
        tag = self.soup.find("i", class_="icon-clock-o")
        if tag and tag.parent:
            raw = tag.parent.get_text(strip=True)
            raw = raw.replace("입력", "").strip()
            try:
                date = datetime.strptime(raw, "%Y.%m.%d %H:%M") # 2025.11.17 07:47
                return date.strftime("%Y-%m-%d")
            except ValueError:
                return None
        return None
         
    def get_contents(self):
        spans = self.soup.find_all("span", style = "font-size:18px;")
        if spans:
            contents = "\n".join(i.get_text(strip = True) for i in spans)
            return contents.split('@')[0].strip()
        
        articles = self.soup.find("article", id = "article-view-content-div")
        if articles:
            ps = articles.find_all("p")
            if ps:
                contents = "\n".join(p.get_text(strip=True) for p in ps)
                return contents.split("@")[0].strip()
                
        justify_ps = self.soup.find_all("p", style=lambda v: v and "text-align: justify" in v)
        if justify_ps:
            contents = "\n".join(p.get_text(strip=True) for p in justify_ps)
            return contents.split("@")[0].strip()
    
        return None
            
    def to_dict(self):
        return {
        
        "id": self.get_id(),
        "title": self.get_title(),
        "contents": self.get_contents(),
        "source": self.get_source(),
        "section": self.get_section(),
        "author": self.get_author(),
        "publish_date": self.get_publish_date(),
        "url": self.get_article_url(),
        "category": self.get_category(),

        }
    
    def __str__(self):
        data = self.to_dict()
        return "\n".join(f"{k}: {v}" for k, v in data.items())

class SPNCrawler:
    def __init__(self, start, end, filename, section=None, sleep_min=0.5, sleep_max=1.0):
        self.start = start
        self.end = end
        self.filename = filename
        self.section = section
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self.existing_ids = self._load_existing_ids()

    def _load_existing_ids(self):
        ids = set()
        if os.path.exists(self.filename):
            with open(self.filename, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "id" in row and row["id"]:
                        ids.add(row["id"])
        return ids

    def _open_writer(self):
        file_exists = os.path.exists(self.filename)
        f = open(self.filename, "a", encoding="utf-8-sig", newline="")
        fieldnames = [
            "id",
            "title",
            "contents",
            "source",
            "section",
            "author",
            "publish_date",
            "url",       
            "category",            
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        return f, writer

    def run(self):
        f, writer = self._open_writer()

        try:
            for i in range(self.start, self.end - 1, -1):
                url = BASE_URL.format(i)

                try:
                    crawler = ArticleCrawler(url)
                    data = crawler.to_dict()

                    # id 없으면 스킵
                    article_id = data.get("id")
                    if not article_id:
                        print(f"[SKIP] idx {i}: id 없음")
                        continue

                    # 이미 저장된 id면 스킵 (재실행 대비)
                    if article_id in self.existing_ids:
                        print(f"[SKIP] idx {i}: 이미 저장된 기사 ({article_id})")
                        continue

                    if self.section and data.get("section") != self.section:
                        print(f"[SKIP] idx {i}: section={data.get('section')}")
                        continue

                    # CSV 추가
                    writer.writerow(data)
                    f.flush()  
                    self.existing_ids.add(article_id)

                    print(f"[SAVE] idx {i}: {article_id} 저장 완료")

                except Exception as e:
                    print(f"[ERROR] idx {i} 에러 발생: {e}")

                # 서버 부하/차단 방지를 위한 랜덤 슬립
                time.sleep(random.uniform(self.sleep_min, self.sleep_max))

        finally:
            f.close()