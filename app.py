
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Steam Review Analytics Dashboard",
    layout="wide"
)

st.title("Steam Review Analytics Dashboard")
st.caption("NLP, Sentiment Analysis, Recommendation System, and Business Insights")

DATA_DIR = Path("dashboard_data")


def load_csv_folder(folder_name):
    folder_path = DATA_DIR / folder_name

    if not folder_path.exists():
        return pd.DataFrame()

    csv_files = list(folder_path.glob("*.csv"))

    if not csv_files:
        return pd.DataFrame()

    return pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)


sentiment_summary = load_csv_folder("sentiment_summary")
player_engagement = load_csv_folder("player_engagement")
keywords = load_csv_folder("keywords")
topic_summary = load_csv_folder("topic_summary")
recommendations = load_csv_folder("recommendations")

page = st.sidebar.radio(
    "Dashboard Navigation",
    [
        "Executive Overview",
        "Sentiment Analysis",
        "Recommendations",
        "Business Insights"
    ]
)

if page == "Executive Overview":
    st.header("Executive Overview")

    total_reviews = 0
    if not sentiment_summary.empty:
        sentiment_summary["review_count"] = sentiment_summary["review_count"].astype(int)
        total_reviews = sentiment_summary["review_count"].sum()

    total_games = 0
    if not player_engagement.empty:
        total_games = player_engagement["app_name"].nunique()

    col1, col2 = st.columns(2)
    col1.metric("Total Reviews Analyzed", f"{total_reviews:,}")
    col2.metric("Total Games Analyzed", f"{total_games:,}")

    st.subheader("Overall Sentiment Distribution")

    if not sentiment_summary.empty:
        fig = px.pie(
            sentiment_summary,
            names="sentiment",
            values="review_count",
            title="Positive vs Negative Review Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sentiment summary data is not available.")

    st.subheader("Most Reviewed Games")

    if not player_engagement.empty:
        player_engagement["review_count"] = player_engagement["review_count"].astype(int)
        top_games = player_engagement.sort_values("review_count", ascending=False).head(10)

        fig = px.bar(
            top_games,
            x="review_count",
            y="app_name",
            orientation="h",
            title="Top 10 Most Reviewed Games"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Player engagement data is not available.")


elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis Page")

    st.subheader("Frequently Occurring Keywords")

    if not keywords.empty:
        keywords["frequency"] = keywords["frequency"].astype(int)
        top_keywords = keywords.sort_values("frequency", ascending=False).head(20)

        fig = px.bar(
            top_keywords,
            x="frequency",
            y="keyword",
            orientation="h",
            title="Top 20 Keywords in Player Reviews"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Keyword data is not available.")

    st.subheader("Common Themes Discussed by Players")

    if not topic_summary.empty:
        topic_summary["review_count"] = topic_summary["review_count"].astype(int)

        fig = px.bar(
            topic_summary,
            x="review_count",
            y="topic",
            orientation="h",
            title="Most Common Discussion Topics"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Topic summary data is not available.")


elif page == "Recommendations":
    st.header("Recommendation Page")

    st.write("Users can view recommended games generated from the content-based recommendation system.")

    if not recommendations.empty:
        st.dataframe(recommendations, use_container_width=True)
    else:
        st.info("Recommendation data is not available.")


elif page == "Business Insights":
    st.header("Business Insights Page")

    st.markdown(
        '''
        ### Recommendations for Game Publishers

        - Address technical issues such as bugs, crashes, lag, and server instability.
        - Prioritize features valued by players, including gameplay, fun, story, and multiplayer experience.
        - Use sentiment trends to monitor player satisfaction over time.
        - Promote highly reviewed and positively rated games to improve player engagement.
        - Use recommendation outputs to increase retention and game discovery.
        '''
    )
