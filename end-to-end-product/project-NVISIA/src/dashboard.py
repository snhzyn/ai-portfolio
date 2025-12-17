# =========================
# import
# =========================
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import folium
import hashlib

from rec import Recommender
from geocoder import Geocoder
from llmtodb import LLMtoDatabase
from knowledge import KnowledgeGraph

import psycopg2
from psycopg2.extras import RealDictCursor


# =========================


notes = """

동작 흐름:
    1) 기사 원문 CSV 파일 업로드 (필수 컬럼: title, content, publish_date, url)
    2) LLMtoDatabase 모듈을 통해
       - 기사 요약(summary)
       - 임베딩 생성(embedding)
       - 카테고리 분류(categorization) 수행
    3) 처리된 데이터를 PostgreSQL 데이터베이스에 저장
    4) Recommender 모듈을 통해 관련 기사 Top-k 추천 (기본값: k = 10)
    5) 추천 
    5) Geocoder 모듈을 통해 기사 관련 위치를 지도에 시각화

"""


# =========================
# DB 설정
# =========================
DB = dict(
    host="localhost",
    database="nvisiaDb",
    user="postgres",
    password="postgres1202",
    port=5432,
)


# =========================
# 파일 경로 세팅 
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]   
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


# =========================
# 공용 커넥터 / 헬퍼
# =========================
def get_psql_conn():
    """간단 쿼리용 psycopg2 커넥션 (글로벌 캐시 X, 매번 열고 닫기)"""
    conn = psycopg2.connect(
        host=DB["host"],
        database=DB["database"],
        user=DB["user"],
        password=DB["password"],
        port=DB["port"],
        options="-c client_encoding=UTF8 -c lc_messages=C",
    )
    return conn

@st.cache_data
def load_all_articles():
    conn = get_psql_conn()
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
    cur.close()
    conn.close()

    df = pd.DataFrame(rows)
    if "publish_date" in df.columns:
        df["publish_date"] = df["publish_date"].astype(str).str[:10]
    return df

def split_event_locs(event_loc_str: str):
    """'평양시, 함경북도 청진시' → ['평양시', '함경북도 청진시']"""
    if not event_loc_str:
        return []
    return [p.strip() for p in event_loc_str.split(",") if p.strip()]

def split_event_orgs(event_org_str: str):
    """selected_id의 event_org를 가져와서 좌표 테이블과 비교. 존재하는 경우 지도에 표시"""
    if not event_org_str:
        return []
    return [o.strip() for o in event_org_str.split(",") if o.strip()]


# =========================
# Streamlit 세팅 (Home -> Dashboard)
# =========================
st.set_page_config(page_title="NVISIA", layout="wide")

# Matplotlib 한글 폰트
plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)

# LLMtoDatabase용 pickle 경로
TFIDF_VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label.pkl"

if "page" not in st.session_state:
    st.session_state["page"] = "home" 
if "uploaded_csv" not in st.session_state:
    st.session_state["uploaded_csv"] = None  
if "ingest_status" not in st.session_state:
    st.session_state["ingest_status"] = {"done": False, "msg": ""}


def go_dashboard():
    st.session_state["page"] = "dashboard"

def go_home():
    st.session_state["page"] = "home"

def render_home():
    st.title("NVISIA: North-Korea Vision & Insights by SIA")

    uploaded = st.file_uploader(
        "csv 파일을 올려주세요. 'title, content, publish_date, url'을 header로 꼭 사용해주세요.", 
        type=["csv"],
        accept_multiple_files=False
    )

    try:
        with open(DATA_DIR / "upload_template.csv", "rb") as f:
            template_bytes = f.read()

        st.download_button(
            label="CSV 템플릿 다운로드",
            data=template_bytes,
            file_name="upload_template.csv",
            mime="text/csv",
            use_container_width=False,
        )

    except FileNotFoundError:
        st.warning(
            "CSV 템플릿 파일을 찾을 수 없습니다. "
            "https://github.com/milkpotato1000/NVISIA 의 data 폴더를 확인해주세요."
        )

    if uploaded is not None:
        st.session_state["uploaded_csv"] = {
            "name": uploaded.name,
            "bytes": uploaded.getvalue(),
        }
        st.success(f"하단의 '기사 업로드 시작'을 눌러 작업을 시작해주세요. 기사의 양에 따라 오랜 시간이 소요될 수도 있습니다.")

    st.markdown("")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button(
            "기사 업로드 시작",
            disabled=(st.session_state["uploaded_csv"] is None),
            use_container_width=True
        ):
            try:
                file_bytes = st.session_state["uploaded_csv"]["bytes"]

                with st.spinner("기사 요약 및 DB 저장 중..."):
                    llm_db = LLMtoDatabase(
                        host=DB["host"],
                        database=DB["database"],
                        user=DB["user"],
                        password=DB["password"],
                        port=DB["port"],
                        tfidf_vectorizer_path=TFIDF_VECTORIZER_PATH,
                        svm_model_path=SVM_MODEL_PATH,
                        label_encoder_path=LABEL_ENCODER_PATH,                        
                    )

                    try:
                        result = llm_db.csv_to_db(file_bytes)
                    finally:
                        llm_db.close()

                st.success(
                    f"완료! total={result['total']} / inserted={result['inserted']} "
                    f"/ skipped_existing={result['skipped_existing']} / skipped_empty={result['skipped_empty']} "
                    f"/ failed={result['failed']}"
                )

                load_all_articles.clear()
                go_dashboard()

            except Exception as e:
                st.error(f"CSV 처리 중 오류: {e}")

    with col2:
        st.button(
            "Dashboard로 이동",
            on_click=go_dashboard,
            use_container_width=True
        )

    if st.session_state.get("ingest_status", {}).get("msg"):
        st.caption(st.session_state["ingest_status"]["msg"])

def render_dashboard():
    st.title("NVISIA: North-Korea Vision & Insights by SIA")
    st.button("Home", on_click=go_home)

    # =========================
    # 객체 호출
    # =========================
    @st.cache_resource
    def get_rec():
        return Recommender(**DB)

    @st.cache_resource
    def get_geo():
        return Geocoder(**DB)
    
    rec = get_rec()
    geo = get_geo()

    if "selected_id" not in st.session_state:
        st.session_state["selected_id"] = None

    if "expanded" not in st.session_state:
        st.session_state.expanded = False


    # =========================
    # 데이터 로드
    # =========================
    df = load_all_articles()

    # 차트용 전역 설정 (카테고리 리스트 및 색상 고정)
    if not df.empty and "category" in df.columns:

        # 빈도수 기준 내림차순 정렬 
        global_counts = df["category"].dropna().value_counts()
        all_categories = global_counts.index.tolist()
    
        # 색상 맵 생성 (파랑, 주황, 초록, 보라, 빨강 순)
        # 직접 HEX 코드 지정 또는 tab10에서 추출하여 순서 배치
    
        custom_palette = [
            "#1f77b4", # Blue
            "#ff7f0e", # Orange
            "#2ca02c", # Green
            "#9467bd", # Purple
            "#d62728", # Red
            "#8c564b", # Brown
            "#e377c2", # Pink
            "#7f7f7f", # Gray
            "#bcbd22", # Olive
            "#17becf", # Cyan
        ]
    
        cat_color_map = {cat: custom_palette[i % len(custom_palette)] for i, cat in enumerate(all_categories)}
    else:
        all_categories = []
        cat_color_map = {}


    # =========================
    # 레이아웃
    #   top_left  : 파이차트
    #   top_right : 추천 기사 테이블
    #   bottom_left : 전체 기사 테이블
    #   bottom_right: 지도
    # =========================
    top_left, top_middle, top_right = st.columns([1, 1, 1])

    st.divider()

    def toggle_expanded():
        st.session_state.expanded = not st.session_state.expanded

    st.button(
        "기사 목록 펼치기" if not st.session_state.expanded else "되돌리기",
        on_click=toggle_expanded,
    )

    table_height = 600 if st.session_state.expanded else 250

    bottom_left, bottom_right = st.columns([2, 1])


    # =========================
    # bottom_left: 전체 기사 목록 + 선택 (row 클릭 방식)
    # =========================
    with bottom_left:
        st.subheader("전체 기사 목록")

        df_display = df[['id', 'title', 'summary', 'publish_date', 'category']].copy()
        st.caption(f"총 {len(df_display)}개 기사 - 더 많은 기사를 보려면 스크롤하세요.")

        event = st.dataframe(
            df_display,
            width="stretch",
            height=table_height,
            selection_mode="single-row",
            on_select="rerun",
            key="article_table",
        )

        if event.selection.rows:
            idx = event.selection.rows[0]
            st.session_state["selected_id"] = df_display.iloc[idx]["id"]
        else:
            st.session_state["selected_id"] = None

        # 지도 조작 등으로 리런될 때 event.selection.rows가 비어있을 수 있어
        # else 구문(선택 해제 시 None 처리)을 제거하여 선택 상태를 유지함.

    selected_id = st.session_state.get("selected_id")


    # =========================
    # top: 추천 기사 테이블 + 파이차트 세팅
    # =========================
    rec_list = []
    rec_ids = []
    rec_df_view = pd.DataFrame()

    chart_df = df.copy()
    chart_title = "전체 뉴스 카테고리"

    if selected_id is not None:
        rec_list = rec.get_similar_articles(selected_id, k=10)
        rec_ids = [r["id"] for r in rec_list]

        if rec_list:
            df_rec = df[df["id"].isin(rec_ids)].copy()
            df_rec.set_index("id", inplace=True)

            rows = []

            # 유저 선택 기사는 최상단 고정
            base_row_all = df[df["id"] == selected_id]
            if not base_row_all.empty:
                base_row = base_row_all.iloc[0]

                base_title = (base_row.get("title", "") or "")
                base_summary = (base_row.get("summary", "") or "")

                rows.append(
                    {
                        "id": selected_id,
                        "title": base_title[:50] + ("..." if len(base_title) > 80 else ""),
                        "summary": base_summary[:50] + ("..." if len(base_summary) > 50 else ""),
                        "category": base_row.get("category", ""),
                        "publish_date": base_row.get("publish_date", ""),
                        "url": base_row.get("url", ""),
                    }
                )

            # 추천 기사 (publish_date 기준 내림차순 정렬)
            rec_rows = []
            for r in rec_list:
                rid = r["id"]
                base = df_rec.loc[rid] if rid in df_rec.index else {}

                title = (base.get("title", r.get("title", "")) or "")
                summary = (base.get("summary", "") or "")
                category = base.get("category", r.get("category", ""))
                publish_date = base.get("publish_date", r.get("publish_date", ""))
                url = base.get("url", r.get("url", ""))

                rec_rows.append(
                    {
                        "id": rid,
                        "title": title[:50] + ("..." if len(title) > 80 else ""),
                        "summary": summary[:50] + ("..." if len(summary) > 50 else ""),
                        "category": category,
                        "publish_date": publish_date,
                        "url": url,
                    }
                )
        
            # 날짜 기준 내림차순 정렬
            rec_rows.sort(key=lambda x: x['publish_date'], reverse=True)
        
            # 합치기
            rows.extend(rec_rows)

            rec_df_view = pd.DataFrame(rows)

            if not rec_df_view.empty:
                chart_df = rec_df_view
                chart_title = "추천 뉴스 카테고리"


    # =========================
    # top_right: 추천 기사 테이블
    # =========================
    with top_right:
        if selected_id is not None:
            st.subheader(f"관련 추천 뉴스 (기준: {selected_id})")
        else:
            st.subheader("관련 추천 뉴스")

        if not rec_df_view.empty:

            column_config = {}
            if 'url' in rec_df_view.columns:
                column_config["url"] = st.column_config.LinkColumn(
                    "Link",
                    display_text="Open Article"
                )

            # 표시할 컬럼 선택
            display_df = rec_df_view[['url', 'id', 'title', 'publish_date']]

            # 스타일링 함수: selected_id와 일치하는 행 강조
            def highlight_row(row):
                if row['id'] == selected_id:
                    # Streamlit 기본 선택 색상과 유사한 붉은 계열의 반투명 배경색 적용
                    return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row)
                return [''] * len(row)

            # 스타일 적용
            styled_df = display_df.style.apply(highlight_row, axis=1)
   
            st.dataframe(
                styled_df,
                width="stretch",
                hide_index=True,
                height=300,
                column_config=column_config,
            )
        else:
            if selected_id is None:
                st.info("아래 목록에서 기사를 선택하면 추천 뉴스가 표시됩니다.")
            else:
                st.info("추천 기사가 없습니다.")


    # =========================
    # top_middle: knowledge graph
    # =========================
    with top_middle:
        st.subheader("Knowledge Graph")

        # 세션 상태 초기화
        if "last_selected_id_for_kg" not in st.session_state:
            st.session_state["last_selected_id_for_kg"] = None
        if "knowledge_fig" not in st.session_state:
            st.session_state["knowledge_fig"] = None
        if "kg_error" not in st.session_state:
            st.session_state["kg_error"] = None

        # 선택된 기사가 변경되었을 때만 그래프 재생성
        if selected_id != st.session_state["last_selected_id_for_kg"]:
            st.session_state["knowledge_fig"] = None
            st.session_state["kg_error"] = None
        
            if rec_list:
                try:
                    know = KnowledgeGraph(rec_list)
                    fig = know.get_figure()
                    st.session_state["knowledge_fig"] = fig
                except Exception as e:
                    st.session_state["kg_error"] = str(e)
        
            st.session_state["last_selected_id_for_kg"] = selected_id

        # 그래프 출력
        if st.session_state["kg_error"]:
            st.error(f"그래프 생성 중 오류가 발생했습니다: {st.session_state['kg_error']}")
        elif st.session_state["knowledge_fig"]:
            st.pyplot(st.session_state["knowledge_fig"], width="content")
        else:
            st.info("추천 기사의 키워드들을 바탕으로 그래프가 생성됩니다.")

    # =========================
    # top_left: 파이차트
    # =========================
    with top_left:
        if "category" in chart_df.columns:
            st.subheader(chart_title)
            category_counts = chart_df["category"].value_counts()

            if not category_counts.empty:

                # 가로 바 차트 (Horizontal Bar Chart)
                # 모든 항목을 항상 y축에 표시. 정렬 기준: 기사 많은 순(all_categories)이 위로 오도록.
                # barh는 y축 0(아래)부터 그리므로, 리스트를 역순([::-1])으로 주어야 '많은 것'이 y축 상단에 위치함.
                y_cats = all_categories[::-1]
            
                # 현재 데이터(chart_df)의 카운트 집계
                current_counts_dict = category_counts.to_dict()
                y_values = [current_counts_dict.get(c, 0) for c in y_cats]
                y_colors = [cat_color_map.get(c, "gray") for c in y_cats]
            
                total = sum(y_values)

                # 차트 크기: 카테고리 개수에 따라 유동적 조절 (기본 4, 항목당 0.3 추가)
                fig_height = max(4.0, len(y_cats) * 0.4)
                fig, ax = plt.subplots(figsize=(5, fig_height))
            
                bars = ax.barh(y_cats, y_values, color=y_colors, height=0.6)

                # 값 표시
                max_val = max(y_values) if y_values else 0
            
                for bar, val in zip(bars, y_values):
                    if val > 0:
                        # 개수 (바 끝)
                        width = bar.get_width()
                        y_pos = bar.get_y() + bar.get_height() / 2
                        ax.text(width, y_pos, f" {int(val)}", va='center', ha='left', fontsize=9)
                    
                        # 비율 (바 내부)
                        if total > 0:
                            pct = (val / total) * 100
                            # 바가 너무 작아서 글씨가 안 들어갈 정도가 아니면 표시
                            # 텍스트가 바 밖으로 나가지 않도록 바 길이의 중간에 표시
                            if width > max_val * 0.1: # 가시성을 위해 일정 길이 이상일 때만 표시
                                ax.text(width / 2, y_pos, f"{pct:.1f}%", va='center', ha='center', 
                                        fontsize=8, color='white', fontweight='bold')

                ax.tick_params(axis='y', labelsize=10)
                ax.tick_params(axis='x', labelsize=9)
            
                # x축 범위 넉넉하게 (텍스트 잘림 방지)
                if max_val > 0:
                    ax.set_xlim(0, max_val * 1.15)
            
                # 상단/우측 테두리 제거로 깔끔하게
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
                st.pyplot(fig, width="content")
            else:
                st.info("입력된 데이터가 없습니다. 데이터를 추가해주세요.")
        else:
            st.info("카테고리 로드 중 오류가 발생했습니다.")


    # =========================
    # bottom_right: 지도 (유저가 선택한 기사 한 건만 표시)
    # =========================
    with bottom_right:

        if selected_id:
            df_sel = df[df["id"] == selected_id].copy()

            # event_loc layer
            locs = set()
            for loc_str in df_sel["event_loc"].fillna(""):
                for loc in split_event_locs(loc_str):
                    locs.add(loc)
            locs = sorted(locs)

            if locs:
                geo_dict = geo.get_geometry(locs)

                center = (39.0, 127.0)
                m = folium.Map(location=center, zoom_start=7)

                def color_from_name(name):
                    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
                    return f"#{h}"

                for loc, geom in geo_dict.items():
                    feature = {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {"event_loc": loc},
                    }

                    folium.GeoJson(
                        feature,
                        name=loc,
                        tooltip=folium.Tooltip(loc),
                        style_function=lambda x, loc_name=loc: {
                            "fillColor": color_from_name(loc_name),
                            "color": "black",
                            "weight": 1,
                            "fillOpacity": 0.6,
                        },
                        highlight_function=lambda x: {
                            "weight": 3,
                            "color": "yellow",
                            "fillOpacity": 0.8,
                        },
                    ).add_to(m)

                # event_org layer
                event_orgs = set()
                for org_str in df_sel["event_org"].fillna(""):
                    for org in split_event_orgs(org_str):
                        event_orgs.add(org)
                event_orgs = sorted(event_orgs)

                org_rows = geo.do_spatial_join(locs, event_orgs)

                org_fg = folium.FeatureGroup(
                    name = "주요 위치",
                    show = True
                )

                if org_rows:
                    for r in org_rows:
                        folium.CircleMarker(
                            location=[r["y_4326"], r["x_4326"]],
                            radius=3,
                            tooltip=r["org_name"],
                            fill=True,
                            fill_opacity=0.9,
                            color="red",
                        ).add_to(org_fg)

                org_fg.add_to(m)

                folium.LayerControl(collapsed=False).add_to(m)

                left_spacer, center_col, right_spacer = st.columns([0.5, 3, 0.5])
                with center_col:
                    st_folium(m, width="100%", height=400)

            else:
                st.info("선택된 기사에 위치 정보가 없습니다.")
        else:
            st.info("위치를 조회하고자 하는 기사를 선택해주세요.")

# =========================
# Router 실행
# =========================
if st.session_state["page"] == "home":
    render_home()
else:
    render_dashboard()