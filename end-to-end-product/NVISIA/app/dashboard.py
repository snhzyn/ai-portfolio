import pandas as pd

import streamlit as st
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

from core.config import DB, DATA_DIR

from app.services.article_reader import ArticleReader
from app.services.chart_service import prepare_category_chart_context, build_category_bar_chart
from app.services.ingestion_runner import run_csv_ingestion
from app.services.knowledge import KnowledgeGraph
from app.services.recommendation_view import build_recommendation_view
from app.services.resources import get_recommender, get_geocoder
from app.services.session_state import init_session_state

# =========================
# Streamlit 세팅 (Home -> Dashboard)
# =========================
st.set_page_config(page_title="NVISIA", layout="wide")

# Matplotlib 한글 폰트
plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)

# session 초기화
init_session_state(st.session_state)

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
                    result = run_csv_ingestion(file_bytes)

                st.success(
                    f"저장이 완료되었습니다. 전체 기사={result['total']} / 정상 저장={result['inserted']} "
                    f"/ 중복 기사={result['skipped_existing']} / 누락 기사={result['skipped_empty']} "
                    f"/ 저장 실패={result['failed']}"
                )

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

    rec = get_recommender()
    geo = get_geocoder()


    # =========================
    # 데이터 로드
    # =========================
    reader = ArticleReader(DB)
    df = reader.load_all_articles()

    if df is None:
        st.warning("테이블이 존재하지 않습니다. Home에서 CSV 업로드를 먼저 진행해주세요.")
        st.button("Home으로 이동하기", on_click=go_home)
        st.stop()

    if df.empty:
        st.warning("조회할 기사가 없습니다. Home에서 기사 업로드를 진행해주세요.")
        st.button("Home으로 이동하기", on_click=go_home)
        st.stop()        

    all_categories, cat_color_map = prepare_category_chart_context(df)


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
    rec_df_view = pd.DataFrame()

    chart_df = df.copy()
    chart_title = "전체 뉴스 카테고리"

    if selected_id is not None:
        rec_list = rec.get_similar_articles(selected_id, k=10)
        rec_df_view = build_recommendation_view(df, selected_id, rec_list)

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
        st.subheader(chart_title)

        fig = build_category_bar_chart(chart_df, all_categories, cat_color_map)

        if fig is not None:
            st.pyplot(fig, width="content")
        else:
            if "category" in chart_df.columns:
                st.info("입력된 데이터가 없습니다. 데이터를 추가해주세요.")
            else:
                st.info("카테고리 로드 중 오류가 발생했습니다.")

    # =========================
    # bottom_right: 지도 (유저가 선택한 기사 한 건만 표시)
    # =========================
    with bottom_right:

        if selected_id:
            df_sel = df[df["id"] == selected_id].copy()

            m = geo.build_article_map(df_sel)

            if m is not None:
                _, center_col, _ = st.columns([0.5, 3, 0.5])
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