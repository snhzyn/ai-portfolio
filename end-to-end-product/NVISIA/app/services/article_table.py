import streamlit as st

def render_article_table(df, table_height):

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
        return df_display.iloc[idx]["id"]

    return None