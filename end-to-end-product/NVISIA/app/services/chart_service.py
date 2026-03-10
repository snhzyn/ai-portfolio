import matplotlib.pyplot as plt

CUSTOM_PALETTE = [
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

def prepare_category_chart_context(df):
    """
    Prepare category ordering and color mapping for dashboard charts.
    """
    if df.empty or "category" not in df.columns:
        return [], {}

    global_counts = df["category"].dropna().value_counts()
    all_categories = global_counts.index.tolist()
    cat_color_map = {
        cat: CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)]
        for i, cat in enumerate(all_categories)
    }
    return all_categories, cat_color_map

def build_category_bar_chart(chart_df, all_categories, cat_color_map):
    """
    Build a horizontal bar chart for article category distribution.
    """
    if chart_df.empty or "category" not in chart_df.columns:
        return None

    category_counts = chart_df["category"].value_counts()
    if category_counts.empty:
        return None

    y_cats = all_categories[::-1]
    current_counts_dict = category_counts.to_dict()
    y_values = [current_counts_dict.get(c, 0) for c in y_cats]
    y_colors = [cat_color_map.get(c, "gray") for c in y_cats]

    total = sum(y_values)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(y_cats, y_values, color=y_colors, height=0.6)

    max_val = max(y_values) if y_values else 0

    for bar, val in zip(bars, y_values):
        if val > 0:
            width = bar.get_width()
            y_pos = bar.get_y() + bar.get_height() / 2
            ax.text(width, y_pos, f" {int(val)}", va="center", ha="left", fontsize=9)

            if total > 0:
                pct = (val / total) * 100
                if width > max_val * 0.1:
                    ax.text(
                        width / 2,
                        y_pos,
                        f"{pct:.1f}%",
                        va="center",
                        ha="center",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )

    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)

    if max_val > 0:
        ax.set_xlim(0, max_val * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig