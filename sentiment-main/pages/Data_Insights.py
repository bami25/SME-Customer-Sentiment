import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logger import setup_logger


Logger = setup_logger(logger_file="sentiments")


st.set_page_config(
    page_title="Customer Review Insights",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------

def normalise_column_name(column: str) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:

    df = pd.read_csv(uploaded_file)

    df.columns = [
        normalise_column_name(column)
        for column in df.columns
    ]

    return df


def validate_data(df: pd.DataFrame):

    required_columns = [
        "product_name",
        "review",
        "sentiment_score",
        "overall",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    return missing_columns


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            dayfirst=True,
        )

        if df["date"].notna().any():

            if "month" not in df.columns:
                df["month"] = df["date"].dt.month_name()

            if "year" not in df.columns:
                df["year"] = df["date"].dt.year

    df["sentiment_score"] = pd.to_numeric(
        df["sentiment_score"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "product_name",
            "sentiment_score",
            "overall",
        ]
    )

    return df


# -----------------------------
# Charts
# -----------------------------

def plot_overall_sentiment(df):

    counts = (
        df["overall"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["Sentiment", "Count"]

    return px.bar(
        counts,
        x="Sentiment",
        y="Count",
        title="Distribution of Customer Sentiment",
        text="Count",
    )


def plot_sentiment_score_distribution(df):

    return px.histogram(
        df,
        x="sentiment_score",
        nbins=5,
        title="Distribution of Sentiment Scores",
        labels={
            "sentiment_score": "Sentiment Score",
        },
    )


def plot_sentiment_by_product(df):

    grouped = (
        df.groupby(
            ["product_name", "overall"]
        )
        .size()
        .reset_index(name="Count")
    )

    return px.bar(
        grouped,
        x="product_name",
        y="Count",
        color="overall",
        title="Sentiment by Product",
        barmode="stack",
        labels={
            "product_name": "Product",
            "overall": "Sentiment",
        },
    )


def plot_monthly_sentiment(df):

    if "month" not in df.columns:
        return None

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    working_df = df.copy()

    working_df["month"] = pd.Categorical(
        working_df["month"],
        categories=month_order,
        ordered=True,
    )

    monthly = (
        working_df
        .groupby("month", observed=False)["sentiment_score"]
        .mean()
        .reset_index()
    )

    return px.line(
        monthly,
        x="month",
        y="sentiment_score",
        markers=True,
        title="Average Sentiment Score by Month",
        labels={
            "month": "Month",
            "sentiment_score": "Average Sentiment Score",
        },
    )


def plot_yearly_sentiment(df):

    if "year" not in df.columns:
        return None

    yearly = (
        df.groupby("year")["sentiment_score"]
        .mean()
        .reset_index()
    )

    return px.line(
        yearly,
        x="year",
        y="sentiment_score",
        markers=True,
        title="Average Sentiment Score by Year",
        labels={
            "year": "Year",
            "sentiment_score": "Average Sentiment Score",
        },
    )


# -----------------------------
# Main App
# -----------------------------

def main():

    st.title("Business Data Insights")

    st.write(
        """
        Explore trends in your analysed customer reviews.

        Upload the CSV downloaded from the **Sentiment Analysis**
        page to begin.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload analysed customer review CSV",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info(
            "Upload a sentiment-analysis output file to see insights."
        )
        return

    try:

        df = load_data(uploaded_file)

        missing_columns = validate_data(df)

        if missing_columns:

            st.error(
                "This file does not appear to be an analysed review file."
            )

            st.write(
                "Missing columns:"
            )

            for column in missing_columns:
                st.write(f"- `{column}`")

            st.info(
                "Run the original review file through the "
                "Sentiment Analysis page first, download the result, "
                "and upload that downloaded file here."
            )

            return

        df = clean_data(df)

        st.success(
            f"Loaded {len(df):,} analysed reviews."
        )

        # -----------------------------
        # Summary metrics
        # -----------------------------

        total_reviews = len(df)

        positive_reviews = (
            df["overall"]
            .eq("Positive")
            .sum()
        )

        neutral_reviews = (
            df["overall"]
            .eq("Neutral")
            .sum()
        )

        negative_reviews = (
            df["overall"]
            .eq("Negative")
            .sum()
        )

        average_score = (
            df["sentiment_score"]
            .mean()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Reviews",
            f"{total_reviews:,}",
        )

        c2.metric(
            "Positive",
            f"{positive_reviews:,}",
            f"{positive_reviews / total_reviews:.1%}"
            if total_reviews
            else None,
        )

        c3.metric(
            "Negative",
            f"{negative_reviews:,}",
            f"{negative_reviews / total_reviews:.1%}"
            if total_reviews
            else None,
        )

        c4.metric(
            "Average Score",
            f"{average_score:.2f}/5",
        )

        st.divider()

        # -----------------------------
        # Filters
        # -----------------------------

        st.sidebar.header("Filters")

        products = sorted(
            df["product_name"]
            .dropna()
            .unique()
        )

        selected_products = st.sidebar.multiselect(
            "Products",
            products,
        )

        sentiments = [
            sentiment
            for sentiment in [
                "Positive",
                "Neutral",
                "Negative",
            ]
            if sentiment in df["overall"].unique()
        ]

        selected_sentiments = st.sidebar.multiselect(
            "Sentiment",
            sentiments,
            default=sentiments,
        )

        filtered_df = df.copy()

        if selected_products:

            filtered_df = filtered_df[
                filtered_df["product_name"]
                .isin(selected_products)
            ]

        if selected_sentiments:

            filtered_df = filtered_df[
                filtered_df["overall"]
                .isin(selected_sentiments)
            ]

        st.subheader("Review Insights")

        # -----------------------------
        # Charts
        # -----------------------------

        col1, col2 = st.columns(2)

        with col1:

            fig = plot_overall_sentiment(
                filtered_df
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with col2:

            fig = plot_sentiment_score_distribution(
                filtered_df
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.plotly_chart(
            plot_sentiment_by_product(
                filtered_df
            ),
            use_container_width=True,
        )

        monthly_chart = plot_monthly_sentiment(
            filtered_df
        )

        if monthly_chart is not None:

            st.plotly_chart(
                monthly_chart,
                use_container_width=True,
            )

        yearly_chart = plot_yearly_sentiment(
            filtered_df
        )

        if yearly_chart is not None:

            st.plotly_chart(
                yearly_chart,
                use_container_width=True,
            )

        # -----------------------------
        # Product performance summary
        # -----------------------------

        st.subheader("Product Sentiment Summary")

        product_summary = (
            filtered_df
            .pivot_table(
                index="product_name",
                columns="overall",
                values="review",
                aggfunc="count",
                fill_value=0,
            )
        )

        st.dataframe(
            product_summary,
            use_container_width=True,
        )

        # -----------------------------
        # Review explorer
        # -----------------------------

        st.subheader("Review Explorer")

        display_columns = [
            column
            for column in [
                "product_name",
                "review",
                "sentiment_score",
                "overall",
                "date",
            ]
            if column in filtered_df.columns
        ]

        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            height=450,
        )

    except Exception as exc:

        Logger.exception(
            f"Error generating data insights: {exc}"
        )

        st.error(
            "The file could not be processed."
        )

        with st.expander("Technical details"):
            st.code(str(exc))


if __name__ == "__main__":
    main()