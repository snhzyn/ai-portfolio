import pandas as pd

def truncate_text(text, limit):
    """
    Truncate text for table display.
    """
    text = (text or "").strip()
    return text[:limit] + ("..." if len(text) > limit else "")
  
def build_recommendation_view(df, selected_id, rec_list):
    """
    Build a dataframe for displaying the selected article
    and its recommended articles.
    """
    if selected_id is None or not rec_list:
        return pd.DataFrame()

    rec_ids = [r["id"] for r in rec_list]

    df_rec = df[df["id"].isin(rec_ids)].copy()
    if not df_rec.empty:
        df_rec.set_index("id", inplace=True)

    rows = []

    # 유저 선택 기사는 최상단 고정
    base_row_all = df[df["id"] == selected_id]
    if not base_row_all.empty:
        base_row = base_row_all.iloc[0]

        base_title = base_row.get("title", "") or ""
        base_summary = base_row.get("summary", "") or ""

        rows.append(
            {
                "id": selected_id,
                "title": truncate_text(base_title, 50),
                "summary": truncate_text(base_summary, 50),
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
                "title": truncate_text(title, 50),
                "summary": truncate_text(summary, 50),
                "category": category,
                "publish_date": publish_date,
                "url": url,
            }
        )

    # 날짜 기준 내림차순 정렬
    rec_rows.sort(key=lambda x: x["publish_date"], reverse=True)
    rows.extend(rec_rows)

    return pd.DataFrame(rows)