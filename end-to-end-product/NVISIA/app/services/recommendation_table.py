import streamlit as st

def render_recommendation_table(rec_df_view, selected_id):
    if selected_id is not None:
        st.subheader(f"관련 추천 뉴스 (기준: {selected_id})")
    else:
        st.subheader("관련 추천 뉴스")

    if rec_df_view.empty:
        if selected_id is None:
            st.info("아래 목록에서 기사를 선택하면 추천 뉴스가 표시됩니다.")
        else:
            st.info("추천 기사가 없습니다.")
        return

    column_config = {}
    if "url" in rec_df_view.columns:
        column_config["url"] = st.column_config.LinkColumn(
            "Link",
            display_text="Open Article"
        )

    display_df = rec_df_view[["url", "id", "title", "publish_date"]]

    def highlight_row(row):
        if row["id"] == selected_id:
            return ["background-color: rgba(255, 75, 75, 0.2)"] * len(row)
        return [""] * len(row)

    styled_df = display_df.style.apply(highlight_row, axis=1)

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        height=300,
        column_config=column_config,
    )