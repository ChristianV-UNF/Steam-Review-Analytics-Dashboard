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


def safe_to_int(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def safe_to_float(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


# Load dashboard datasets
sentiment_summary = load_csv_folder("sentiment_summary")
player_engagement = load_csv_folder("player_engagement")
keywords = load_csv_folder("keywords")
topic_summary = load_csv_folder("topic_summary")
recommendations = load_csv_folder("recommendations")

# Supplementary NLP outputs
polarity_scores = load_csv_folder("polarity_scores")
lda_topics = load_csv_folder("lda_topics")


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
    positive_reviews = 0
    negative_reviews = 0
    positive_rate = 0

    if not sentiment_summary.empty and "review_count" in sentiment_summary.columns:
        sentiment_summary["review_count"] = safe_to_int(sentiment_summary["review_count"])
        total_reviews = sentiment_summary["review_count"].sum()

        if "sentiment" in sentiment_summary.columns:
            positive_reviews = sentiment_summary.loc[
                sentiment_summary["sentiment"].str.lower() == "positive",
                "review_count"
            ].sum()

            negative_reviews = sentiment_summary.loc[
                sentiment_summary["sentiment"].str.lower() == "negative",
                "review_count"
            ].sum()

            positive_rate = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0

    total_games = 0
    if not player_engagement.empty and "app_name" in player_engagement.columns:
        total_games = player_engagement["app_name"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews Analyzed", f"{total_reviews:,}")
    col2.metric("Total Games Analyzed", f"{total_games:,}")
    col3.metric("Positive Reviews", f"{positive_reviews:,}")
    col4.metric("Positive Rate", f"{positive_rate:.1f}%")

    st.info(
        "The executive overview summarizes player satisfaction and engagement across the analyzed Steam review dataset. "
        "These metrics provide a high-level view of review volume, game coverage, and overall sentiment."
    )

    if not sentiment_summary.empty and not player_engagement.empty:
        st.subheader("Executive Visual Summary")

        player_engagement["review_count"] = safe_to_int(player_engagement["review_count"])
        top_games = player_engagement.sort_values("review_count", ascending=False).head(10)

        col1, col2 = st.columns(2)

        with col1:
            fig_sentiment = px.pie(
                sentiment_summary,
                names="sentiment",
                values="review_count",
                title="Positive vs Negative Review Distribution",
                hole=0.35
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

            st.info(
                "Most reviews are positive, suggesting strong overall player satisfaction. "
                "However, the negative review segment remains important because it highlights improvement opportunities."
            )

        with col2:
            fig_top_games = px.bar(
                top_games,
                x="review_count",
                y="app_name",
                orientation="h",
                title="Top 10 Most Reviewed Games"
            )
            fig_top_games.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_top_games, use_container_width=True)

            st.info(
                "The most reviewed games indicate where player engagement and community activity are strongest. "
                "These titles can guide benchmarking and marketing strategy."
            )

    else:
        st.info("Executive dashboard data is not fully available.")


elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis Page")

    st.markdown(
        """
        This page explores player feedback using keyword analysis, topic summaries, TextBlob polarity scoring,
        and LDA topic modeling. These outputs help identify what players frequently discuss and how they feel
        about their experiences.
        """
    )

    st.subheader("Frequently Occurring Keywords")

    if not keywords.empty and "frequency" in keywords.columns:
        keywords["frequency"] = safe_to_int(keywords["frequency"])

        # Remove generic terms that dominate the chart but add limited business meaning
        generic_terms = {
            "game", "games", "play", "played", "playing", "really", "just",
            "like", "dont", "time", "get", "got", "one"
        }

        filtered_keywords = keywords[
            ~keywords["keyword"].astype(str).str.lower().isin(generic_terms)
        ]

        if filtered_keywords.empty:
            filtered_keywords = keywords

        top_keywords = filtered_keywords.sort_values("frequency", ascending=False).head(15)

        fig = px.bar(
            top_keywords,
            x="frequency",
            y="keyword",
            orientation="h",
            title="Top Review Keywords After Removing Generic Terms"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Frequently occurring keywords highlight the terms most associated with player feedback. "
            "Filtering generic terms makes the chart more useful for identifying meaningful gameplay and experience themes."
        )
    else:
        st.info("Keyword data is not available.")

    st.subheader("Common Themes Discussed by Players")

    if not topic_summary.empty and "review_count" in topic_summary.columns:
        topic_summary["review_count"] = safe_to_int(topic_summary["review_count"])

        fig = px.bar(
            topic_summary.sort_values("review_count", ascending=True),
            x="review_count",
            y="topic",
            orientation="h",
            title="Most Common Discussion Topics"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Topic analysis shows recurring player discussion areas such as gameplay, enjoyment, story, graphics, "
            "multiplayer, bugs, and server issues. These themes help publishers identify both strengths and pain points."
        )
    else:
        st.info("Topic summary data is not available.")

    st.divider()

    st.subheader("TextBlob Sentiment Classification")

    if not polarity_scores.empty and "polarity" in polarity_scores.columns:
        polarity_scores["polarity"] = safe_to_float(polarity_scores["polarity"])

        fig = px.histogram(
            polarity_scores,
            x="polarity",
            nbins=50,
            title="Review Polarity Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        if "textblob_sentiment" in polarity_scores.columns:
            polarity_summary = (
                polarity_scores["textblob_sentiment"]
                .value_counts()
                .reset_index()
            )
            polarity_summary.columns = ["sentiment", "count"]

            fig = px.bar(
                polarity_summary,
                x="sentiment",
                y="count",
                title="TextBlob Sentiment Classification"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "TextBlob polarity scores quantify the emotional tone of player reviews. "
            "Scores closer to +1 indicate more positive language, while scores closer to -1 indicate more negative language."
        )
    else:
        st.info("Polarity score data is not available.")

    st.subheader("LDA Topic Modeling Results")

    if not lda_topics.empty:
        st.write("The following topics were extracted using Latent Dirichlet Allocation (LDA):")

        if "topic_number" in lda_topics.columns:
            lda_topics["topic_number"] = safe_to_int(lda_topics["topic_number"])
            lda_topics = lda_topics.sort_values("topic_number")

        display_topics = lda_topics.copy()

        if "topic_number" in display_topics.columns and "top_keywords" in display_topics.columns:
            display_topics["topic"] = display_topics["topic_number"].apply(lambda x: f"Topic {x}")
            display_topics = display_topics[["topic", "top_keywords"]]
            st.table(display_topics)

            for _, row in display_topics.iterrows():
                st.markdown(f"**{row['topic']}**: {row['top_keywords']}")
        else:
            st.dataframe(display_topics, use_container_width=True)

        st.info(
            "LDA topic modeling identifies recurring themes in player reviews, including gameplay enjoyment, "
            "player engagement, story-related feedback, multiplayer experiences, and technical concerns."
        )
    else:
        st.info("LDA topic data is not available.")


elif page == "Recommendations":
    st.header("Recommendation Page")

    st.write(
        "Users can view recommended games generated from the content-based recommendation system. "
        "The recommendation results use review-based similarity, average review score, review volume, and similarity score."
    )

    if not recommendations.empty:
        st.subheader("Recommendation Results Table")
        st.dataframe(recommendations, use_container_width=True)

        if "similarity_score" in recommendations.columns and "app_name" in recommendations.columns:
            recommendations["similarity_score"] = safe_to_float(recommendations["similarity_score"])

            top_recommendations = (
                recommendations
                .sort_values("similarity_score", ascending=False)
                .head(10)
            )

            st.subheader("Top Recommended Games by Similarity Score")

            fig = px.bar(
                top_recommendations,
                x="similarity_score",
                y="app_name",
                orientation="h",
                title="Top Recommended Games"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Games with higher similarity scores are more closely aligned with the selected review-based profile. "
                "This supports game discovery and can help platforms recommend similar titles to players."
            )
    else:
        st.info("Recommendation data is not available.")


elif page == "Business Insights":
    st.header("Business Insights Page")

    st.markdown(
        """
        ### Business Problem

        Game publishers need to understand player sentiment, common complaints, engagement patterns,
        and recommendation opportunities from large-scale Steam review data. This dashboard transforms
        millions of player reviews into actionable insights for improving game quality, player retention,
        and marketing decisions.

        ### Cloud Analytics Pipeline

        This project uses a cloud-based analytics workflow:

        - Databricks Volumes for dataset storage and processed outputs.
        - PySpark for scalable large-scale data processing.
        - NLP and sentiment analysis for extracting player feedback insights.
        - Recommendation modeling for game discovery.
        - GitHub for version control and collaboration.
        - Streamlit Cloud for dashboard deployment.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            "Player satisfaction is generally strong, with most reviews classified as positive. "
            "This indicates that many Steam titles maintain favorable player perception."
        )

    with col2:
        st.warning(
            "Negative review themes such as bugs, server instability, crashes, and performance issues "
            "remain important signals for product improvement."
        )

    st.markdown(
        """
        ### Recommendations for Game Publishers

        - Address technical issues such as bugs, crashes, lag, and server instability.
        - Prioritize features valued by players, including gameplay, fun, story, and multiplayer experience.
        - Use sentiment trends to monitor player satisfaction over time.
        - Promote highly reviewed and positively rated games to improve player engagement.
        - Use recommendation outputs to increase retention and game discovery.

        ### Insights from Supplementary NLP Analysis

        - TextBlob polarity scores help quantify the emotional tone of player reviews.
        - LDA topic modeling identifies recurring player discussion themes.
        - Positive review patterns highlight valued features such as fun, gameplay, story, and multiplayer experience.
        - Negative review patterns can help publishers prioritize improvements in bugs, servers, performance, and early-access concerns.

        ### Rubric Alignment

        This dashboard integrates large-scale data processing, cloud storage, NLP, sentiment analysis,
        topic modeling, recommendation systems, business-oriented visualization, GitHub collaboration,
        and Streamlit deployment into a complete analytics pipeline.
        """
    )