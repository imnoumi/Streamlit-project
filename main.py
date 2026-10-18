"""
Streamlit Power Demo
====================
A single-file tour of Streamlit's core capabilities: widgets, layouts,
native charts, and integrations with Matplotlib, Plotly, and Altair.

Run manually after installing dependencies:
    .venv/Scripts/python.exe -m pip install -r requirements.txt
    .venv/Scripts/python.exe -m streamlit run main.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import altair as alt


# ---------------------------------------------------------------------------
# Page config & global settings
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Streamlit Power Demo",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use a consistent theme for matplotlib inside Streamlit
plt.style.use("seaborn-v0_8-darkgrid")


# ---------------------------------------------------------------------------
# Cached data generators (showcase st.cache_data)
# ---------------------------------------------------------------------------
@st.cache_data
def make_time_series(n_points: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_points)
    return pd.DataFrame(
        {
            "t": t,
            "sine": np.sin(t) + rng.normal(0, 0.1, n_points),
            "cosine": np.cos(t) + rng.normal(0, 0.1, n_points),
            "trend": 0.05 * t + rng.normal(0, 0.05, n_points),
        }
    )


@st.cache_data
def make_categories(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    return pd.DataFrame(
        {
            "category": rng.choice(categories, n),
            "value": rng.normal(50, 15, n).clip(0, 100),
            "score": rng.uniform(0, 10, n),
        }
    )


@st.cache_data
def make_geo_points(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lat": rng.uniform(40.6, 40.85, n),
            "lon": rng.uniform(-74.05, -73.85, n),
        }
    )


# ---------------------------------------------------------------------------
# Sidebar: global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Global Controls")
    n_points = st.slider("Time series length", 50, 2000, 500, step=50)
    seed = st.number_input("Random seed", 0, 100, 42, step=1)
    theme_color = st.color_picker("Accent color", "#00d4ff")
    show_raw = st.checkbox("Show raw data tables", value=False)

    st.divider()
    st.markdown(
        "**Tip:** every widget above is reactive — move them and the whole "
        "page recomputes automatically."
    )


# ---------------------------------------------------------------------------
# Header & metrics
# ---------------------------------------------------------------------------
st.title("Streamlit Power Demo")
st.markdown(
    "A tour of what Streamlit can do with a single Python file: "
    "reactive widgets, native charts, and Matplotlib / Plotly / Altair "
    "integrations."
)

ts_df = make_time_series(n_points, int(seed))
cat_df = make_categories(400, int(seed))
geo_df = make_geo_points(200, int(seed))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Time points", f"{len(ts_df):,}", delta=f"seed={seed}")
col2.metric("Categories", f"{cat_df['category'].nunique()}")
col3.metric("Mean value", f"{cat_df['value'].mean():.2f}", delta=f"{cat_df['value'].std():.2f} std")
col4.metric("Geo points", f"{len(geo_df):,}")


# ---------------------------------------------------------------------------
# Tabs: different chart families
# ---------------------------------------------------------------------------
tab_native, tab_mpl, tab_plotly, tab_altair, tab_layout = st.tabs(
    ["Native charts", "Matplotlib", "Plotly", "Altair", "Layout & widgets"]
)


# --- Native Streamlit charts ------------------------------------------------
with tab_native:
    st.subheader("Built-in charts (no extra deps beyond pandas/numpy)")

    st.write("Line chart")
    st.line_chart(ts_df.set_index("t")[["sine", "cosine", "trend"]], height=300)

    left, right = st.columns(2)
    with left:
        st.write("Area chart")
        st.area_chart(ts_df.set_index("t")[["sine", "cosine"]], height=250)
    with right:
        st.write("Bar chart (mean value per category)")
        bar_df = cat_df.groupby("category")["value"].mean().sort_values()
        st.bar_chart(bar_df, height=250)

    st.write("Scatter chart")
    st.scatter_chart(
        cat_df,
        x="value",
        y="score",
        color="category",
        height=300,
    )

    st.write("Map (native st.map)")
    st.map(geo_df, height=350)


# --- Matplotlib -------------------------------------------------------------
with tab_mpl:
    st.subheader("Matplotlib figures via st.pyplot")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ts_df["t"], ts_df["sine"], label="sine", color=theme_color)
    axes[0].plot(ts_df["t"], ts_df["cosine"], label="cosine", color="#ff6b6b")
    axes[0].set_title("Sine vs Cosine with noise")
    axes[0].legend()

    hist_vals, hist_bins, _ = axes[1].hist(
        cat_df["value"], bins=30, color=theme_color, edgecolor="black", alpha=0.8
    )
    axes[1].set_title("Value distribution")
    axes[1].set_xlabel("value")
    axes[1].set_ylabel("count")

    st.pyplot(fig, use_container_width=True)

    st.write("A second standalone matplotlib figure")
    fig2, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(cat_df["value"], cat_df["score"], c=cat_df["score"], cmap="viridis", alpha=0.6)
    ax.set_xlabel("value")
    ax.set_ylabel("score")
    ax.set_title("Scatter colored by score")
    st.pyplot(fig2, use_container_width=True)


# --- Plotly ----------------------------------------------------------------
with tab_plotly:
    st.subheader("Interactive Plotly charts via st.plotly_chart")

    fig_px = px.line(
        ts_df,
        x="t",
        y=["sine", "cosine", "trend"],
        title="Time series (Plotly, hover/zoom enabled)",
        template="plotly_dark",
    )
    st.plotly_chart(fig_px, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig_bar = px.bar(
            cat_df.groupby("category")["value"].mean().reset_index(),
            x="category",
            y="value",
            color="category",
            title="Mean value per category",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with right:
        fig_box = px.box(
            cat_df,
            x="category",
            y="value",
            color="category",
            title="Value distribution per category",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.write("3D scatter")
    fig_3d = px.scatter_3d(
        cat_df,
        x="value",
        y="score",
        z=cat_df.groupby("category")["value"].transform("mean"),
        color="category",
        opacity=0.7,
        title="3D scatter",
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.write("Plotly Graph Object (custom gauge)")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cat_df["value"].mean(),
        title={"text": "Average value"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": theme_color}},
    ))
    st.plotly_chart(gauge, use_container_width=True)


# --- Altair ----------------------------------------------------------------
with tab_altair:
    st.subheader("Declarative Altair charts via st.altair_chart")

    base = alt.Chart(cat_df).encode(
        x="value:Q",
        y="score:Q",
        color="category:N",
        tooltip=["category", "value", "score"],
    )
    scatter = base.mark_circle(opacity=0.6, size=60).properties(
        title="Altair scatter", height=350
    )
    st.altair_chart(scatter, use_container_width=True)

    bar = (
        alt.Chart(cat_df)
        .mark_bar()
        .encode(x="category:N", y="mean(value):Q", color="category:N")
        .properties(title="Altair bar (mean value)", height=300)
    )
    st.altair_chart(bar, use_container_width=True)

    st.write("Layered chart: histogram + rule for mean")
    hist = (
        alt.Chart(cat_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=30)),
            y="count():Q",
        )
        .properties(height=300)
    )
    mean_rule = (
        alt.Chart(cat_df)
        .mark_rule(color="red", strokeWidth=2)
        .encode(x="mean(value):Q")
    )
    st.altair_chart(hist + mean_rule, use_container_width=True)


# --- Layout & widget showcase ----------------------------------------------
with tab_layout:
    st.subheader("Layout primitives & interactive widgets")

    with st.expander("Columns + containers"):
        c1, c2, c3 = st.columns([1, 2, 1])
        c1.markdown("### Narrow")
        c1.write("This column is 1 unit wide.")
        c2.markdown("### Wide")
        c2.line_chart(ts_df.set_index("t")[["sine"]], height=180)
        c3.markdown("### Narrow")
        c3.write("Reactive to sidebar seed.")

    with st.expander("Form with submit button"):
        with st.form("my_form"):
            name = st.text_input("Name", "Ada")
            age = st.slider("Age", 0, 100, 30)
            agree = st.checkbox("Accept terms")
            submitted = st.form_submit_button("Submit")
        if submitted:
            st.success(f"Submitted: {name}, age {age}, agreed={agree}")

    with st.expander("Selectbox / multiselect / radio"):
        option = st.selectbox("Pick a category", cat_df["category"].unique())
        multi = st.multiselect("Pick several", ts_df.columns[1:].tolist(), default=["sine"])
        radio = st.radio("Chart type", ["line", "bar", "area"])
        st.write(f"You picked: {option} / {multi} / {radio}")

        subset = ts_df.set_index("t")[multi]
        if radio == "line":
            st.line_chart(subset, height=250)
        elif radio == "bar":
            st.bar_chart(subset, height=250)
        else:
            st.area_chart(subset, height=250)

    with st.expander("Progress & spinner"):
        if st.button("Run a fake job"):
            bar = st.progress(0.0)
            for i in range(1, 101):
                bar.progress(i / 100.0)
            bar.empty()
            st.balloons()

    with st.expander("Code block & JSON"):
        st.code(
            "import streamlit as st\nst.title('Hello')\nst.write('World')",
            language="python",
        )
        st.json({"seed": int(seed), "n_points": n_points, "categories": list(cat_df["category"].unique())})


# ---------------------------------------------------------------------------
# Optional raw data tables
# ---------------------------------------------------------------------------
if show_raw:
    st.divider()
    st.subheader("Raw data")
    st.dataframe(ts_df, use_container_width=True)
    st.dataframe(cat_df, use_container_width=True)
    st.dataframe(geo_df, use_container_width=True)
    st.data_editor(cat_df.head(10), num_rows="dynamic")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Built with Streamlit — edit a widget above and watch the page react.")