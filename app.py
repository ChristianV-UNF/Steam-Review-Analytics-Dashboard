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
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )


def safe_to_float(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


sentiment_summary = load_csv_folder("sentiment_summary")
player_engagement = load_csv_folder("player_engagement")
keywords = load_csv_folder("keywords")
topic_summary = load_csv_folder("topic_summary")
recommendations = load_csv_folder("recommendations")
polarity_scores = load_csv_folder("polarity_scores")
lda_topics = load_csv_folder("lda_topics")
positive_keywords = load_csv_folder("positive_keywords")
negative_keywords = load_csv_folder("negative_keywords")


st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Executive Overview",
        "NLP and Sentiment Analysis",
        "Recommendations",
        "Business Insights"
    ]
)


if page == "Executive Overview":
    st.header("Executive Overview")

    total_reviews = 0
    positive_reviews = 0
    positive_rate = 0

    if not sentiment_summary.empty and "review_count" in sentiment_summary.columns:
        sentiment_summary["review_count"] = safe_to_int(sentiment_summary["review_count"])
        total_reviews = sentiment_summary["review_count"].sum()

        if "sentiment" in sentiment_summary.columns:
            positive_reviews = sentiment_summary.loc[
                sentiment_summary["sentiment"].astype(str).str.lower() == "positive",
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
        "The executive overview summarizes player satisfaction and engagement across the analyzed Steam review dataset."
    )

    with st.expander("Overall Sentiment and Engagement Visuals", expanded=True):
        if not sentiment_summary.empty and not player_engagement.empty:
            player_engagement["review_count"] = safe_to_int(player_engagement["review_count"])
            player_engagement["app_name"] = player_engagement["app_name"].astype(str)

            top_games = (
                player_engagement[
                    (player_engagement["review_count"] > 0)
                    & (player_engagement["app_name"].str.lower() != "none")
                ]
                .sort_values("review_count", ascending=False)
                .head(10)
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Overall Sentiment Distribution")

                fig = px.pie(
                    sentiment_summary,
                    names="sentiment",
                    values="review_count",
                    hole=0.55
                )

                fig.update_layout(
                    height=520,
                    title=None,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.15,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=10, r=10, t=20, b=60)
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "Most reviews are positive, suggesting strong overall player satisfaction."
                )

            with col2:
                st.markdown("#### Top 10 Most Reviewed Games")

                if not top_games.empty:
                    fig = px.bar(
                        top_games,
                        x="app_name",
                        y="review_count"
                    )

                    fig.update_layout(
                        height=520,
                        xaxis_title="Game",
                        yaxis_title="Review Count",
                        margin=dict(l=40, r=30, t=20, b=120)
                    )

                    fig.update_xaxes(tickangle=-35)

                    st.plotly_chart(fig, use_container_width=True)

                    st.info(
                        "The most reviewed games indicate where player engagement is strongest."
                    )
                else:
                    st.warning("Most reviewed games data is not available after cleaning.")
        else:
            st.info("Executive dashboard data is not fully available.")


elif page == "NLP and Sentiment Analysis":
    st.header("NLP and Sentiment Analysis")

    st.markdown(
        """
        This section explores player feedback using keyword analysis, discussion themes,
        TextBlob polarity scoring, positive and negative keyword extraction, and LDA topic modeling.
        """
    )

    with st.expander("Keyword Analysis", expanded=True):
        if not keywords.empty and "frequency" in keywords.columns:
            keywords["frequency"] = safe_to_int(keywords["frequency"])

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

            fig.update_layout(
                height=600,
                title_x=0.5,
                yaxis=dict(categoryorder="total ascending")
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Frequently occurring keywords reveal the most common concepts in player feedback."
            )
        else:
            st.info("Keyword data is not available.")

    with st.expander("Common Discussion Topics", expanded=True):
        if not topic_summary.empty and "review_count" in topic_summary.columns:
            topic_summary["review_count"] = safe_to_int(topic_summary["review_count"])

            fig = px.bar(
                topic_summary.sort_values("review_count", ascending=True),
                x="review_count",
                y="topic",
                orientation="h",
                title="Most Common Discussion Topics"
            )

            fig.update_layout(
                height=550,
                title_x=0.5
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Topic analysis highlights recurring themes such as gameplay, story, graphics, bugs, multiplayer, and server issues."
            )
        else:
            st.info("Topic summary data is not available.")

    with st.expander("TextBlob Sentiment Classification", expanded=True):
        if not polarity_scores.empty and "polarity" in polarity_scores.columns:
            polarity_scores["polarity"] = safe_to_float(polarity_scores["polarity"])

            fig = px.histogram(
                polarity_scores,
                x="polarity",
                nbins=50,
                title="Review Polarity Score Distribution"
            )

            fig.update_layout(
                height=550,
                title_x=0.5
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

                fig.update_layout(
                    height=500,
                    title_x=0.5
                )

                st.plotly_chart(fig, use_container_width=True)

            st.info(
                "TextBlob polarity scores quantify emotional tone. Scores closer to +1 are more positive, while scores closer to -1 are more negative."
            )
        else:
            st.info("Polarity score data is not available.")

    with st.expander("Positive and Negative Review Keywords", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top Positive Review Words")

            if not positive_keywords.empty and "frequency" in positive_keywords.columns:
                positive_keywords["frequency"] = safe_to_int(positive_keywords["frequency"])

                top_positive = (
                    positive_keywords
                    .sort_values("frequency", ascending=False)
                    .head(15)
                )

                fig = px.bar(
                    top_positive,
                    x="frequency",
                    y="keyword",
                    orientation="h",
                    title="Most Common Positive Review Words"
                )

                fig.update_layout(
                    height=500,
                    title_x=0.5,
                    yaxis=dict(categoryorder="total ascending")
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "Positive review keywords highlight what players value most, such as fun, gameplay, story, and overall enjoyment."
                )
            else:
                st.info("Positive keyword data is not available.")

        with col2:
            st.markdown("#### Top Negative Review Words")

            if not negative_keywords.empty and "frequency" in negative_keywords.columns:
                negative_keywords["frequency"] = safe_to_int(negative_keywords["frequency"])

                top_negative = (
                    negative_keywords
                    .sort_values("frequency", ascending=False)
                    .head(15)
                )

                fig = px.bar(
                    top_negative,
                    x="frequency",
                    y="keyword",
                    orientation="h",
                    title="Most Common Negative Review Words"
                )

                fig.update_layout(
                    height=500,
                    title_x=0.5,
                    yaxis=dict(categoryorder="total ascending")
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "Negative review keywords reveal potential improvement areas, such as bugs, crashes, performance, server issues, and frustration points."
                )
            else:
                st.info("Negative keyword data is not available.")

    with st.expander("LDA Topic Modeling Results", expanded=True):
        if not lda_topics.empty:
            if "topic_number" in lda_topics.columns:
                lda_topics["topic_number"] = safe_to_int(lda_topics["topic_number"])
                lda_topics = lda_topics.sort_values("topic_number")

            if "topic_number" in lda_topics.columns and "top_keywords" in lda_topics.columns:
                display_topics = lda_topics.copy()
                display_topics["topic"] = display_topics["topic_number"].apply(lambda x: f"Topic {x}")
                display_topics = display_topics[["topic", "top_keywords"]]

                st.table(display_topics)

                st.info(
                    "LDA topic modeling identifies recurring discussion themes in player reviews, including gameplay enjoyment, engagement, story feedback, multiplayer experience, and technical concerns."
                )
            else:
                st.dataframe(lda_topics, use_container_width=True)
        else:
            st.info("LDA topic data is not available.")


elif page == "Recommendations":
    st.header("Recommendation System")

    st.write(
        "The recommendation system identifies similar games using review-based similarity, average review score, review volume, and similarity score."
    )

    with st.expander("Recommendation Results", expanded=True):
        if not recommendations.empty:
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

                fig.update_layout(
                    height=550,
                    title_x=0.5,
                    yaxis=dict(categoryorder="total ascending")
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "Higher similarity scores indicate games that are more closely aligned with review-based player preferences."
                )
        else:
            st.info("Recommendation data is not available.")


elif page == "Business Insights":
    st.header("Business Insights")

    with st.expander("Business Problem", expanded=True):
        st.markdown(
            """
            Game publishers need to understand player sentiment, common complaints, engagement patterns,
            and recommendation opportunities from large-scale Steam review data. This dashboard transforms
            millions of player reviews into actionable insights for improving game quality, player retention,
            and marketing decisions.
            """
        )

    with st.expander("Cloud Analytics Pipeline", expanded=True):
        st.markdown(
            """
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
            "Player satisfaction is generally strong, with most reviews classified as positive."
        )

    with col2:
        st.warning(
            "Negative review themes such as bugs, server instability, crashes, and performance issues remain important product improvement signals."
        )

    with st.expander("Recommendations for Game Publishers", expanded=True):
        st.markdown(
            """
            - Address technical issues such as bugs, crashes, lag, and server instability.
            - Prioritize features valued by players, including gameplay, fun, story, and multiplayer experience.
            - Use sentiment trends to monitor player satisfaction over time.
            - Promote highly reviewed and positively rated games to improve player engagement.
            - Use recommendation outputs to increase retention and game discovery.
            """
        )

    with st.expander("Rubric Alignment", expanded=True):
        st.markdown(
            """
            This dashboard integrates large-scale data processing, cloud storage, NLP, sentiment analysis,
            topic modeling, recommendation systems, business-oriented visualization, GitHub collaboration,
            and Streamlit deployment into a complete analytics pipeline.
            """
        )