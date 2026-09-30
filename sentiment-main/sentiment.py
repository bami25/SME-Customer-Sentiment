import datetime
from typing import Tuple

import chardet
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from logger import setup_logger


Logger = setup_logger(logger_file="sentiments")


st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="💬",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------

def normalise_column_name(column: str) -> str:
    """
    Convert column names like:
    'Product Name' -> 'product_name'
    'Sentiment Score' -> 'sentiment_score'
    """
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


@st.cache_resource
def load_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load the Hugging Face sentiment model once and reuse it
    across Streamlit reruns.
    """
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        Logger.info("Sentiment model loaded successfully.")
        return tokenizer, model

    except Exception as exc:
        Logger.error(f"Error loading model: {exc}")
        raise


def read_uploaded_csv(csv_file) -> pd.DataFrame:
    """
    Detect encoding, read uploaded CSV, and standardise column names.
    """
    try:
        raw_data = csv_file.read()

        detected = chardet.detect(raw_data)
        encoding = detected.get("encoding") or "utf-8"

        csv_file.seek(0)

        df = pd.read_csv(csv_file, encoding=encoding)

        df.columns = [normalise_column_name(col) for col in df.columns]

        Logger.info(f"CSV successfully read using encoding: {encoding}")

        return df

    except Exception as exc:
        Logger.error(f"Error reading uploaded CSV: {exc}")
        raise


def validate_columns(df: pd.DataFrame):
    """
    Validate minimum required columns.
    """
    required_columns = ["product_name", "review"]

    missing = [column for column in required_columns if column not in df.columns]

    return missing


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise the date column.

    Supports formats such as:
    2014-07-23
    23-07-14
    23/07/2014
    """
    if "date" not in df.columns:
        return df

    original_date = df["date"].copy()

    # First try general parsing
    parsed = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=True,
    )

    invalid_rows = parsed.isna() & original_date.notna()

    if invalid_rows.any():
        invalid_count = int(invalid_rows.sum())

        raise ValueError(
            f"{invalid_count} date value(s) could not be recognised."
        )

    df["date"] = parsed.dt.strftime("%Y-%m-%d")

    # Generate month/year if missing
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")

    if "month" not in df.columns:
        df["month"] = parsed_dates.dt.month_name()

    if "year" not in df.columns:
        df["year"] = parsed_dates.dt.year

    return df


def sentiment_score(
    review: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
) -> int:
    """
    Generate sentiment score from 1 to 5.
    """
    if not isinstance(review, str) or not review.strip():
        return 0

    try:
        encoded = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            result = model(**encoded)

        score = int(torch.argmax(result.logits, dim=1).item() + 1)

        return score

    except Exception as exc:
        Logger.warning(f"Error calculating sentiment score: {exc}")
        return 0


def map_sentiment(score: int) -> str:
    """
    Convert 1-5 score into a business-friendly sentiment label.
    """
    if score in (4, 5):
        return "Positive"

    if score == 3:
        return "Neutral"

    if score in (1, 2):
        return "Negative"

    return "Unknown"


# -----------------------------
# Main App
# -----------------------------

def main():

    st.title("Customer Review Sentiment Analysis")

    st.write(
        """
        Upload a CSV containing customer reviews and the app will analyse
        each review using a BERT-based sentiment model.

        The minimum required columns are:

        - `Product_Name`
        - `Review`

        Optional columns such as `Date`, `Month`, and `Year` will also be
        processed when available.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload customer review CSV",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        return

    try:

        with st.spinner("Reading and validating file..."):
            df = read_uploaded_csv(uploaded_file)

        missing_columns = validate_columns(df)

        if missing_columns:
            st.error(
                "The uploaded CSV is missing the following required "
                f"column(s): {', '.join(missing_columns)}"
            )

            st.write("Columns detected:")
            st.code(", ".join(df.columns))

            return

        try:
            df = parse_dates(df)

        except ValueError as exc:
            st.error(str(exc))
            return

        st.success(
            f"File loaded successfully — {len(df):,} reviews found."
        )

        st.subheader("Data Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Reviews",
                f"{len(df):,}",
            )

        with col2:
            st.metric(
                "Products",
                (
                    df["product_name"].nunique()
                    if "product_name" in df.columns
                    else "N/A"
                ),
            )

        with col3:
            st.metric(
                "Missing reviews",
                int(df["review"].isna().sum()),
            )

        st.divider()

        if st.button(
            "Analyse Reviews",
            type="primary",
            use_container_width=True,
        ):

            tokenizer, model = load_model()

            reviews = df["review"].fillna("").astype(str).tolist()

            scores = []

            progress_bar = st.progress(0)
            status = st.empty()

            total_reviews = len(reviews)

            for index, review in enumerate(reviews):

                score = sentiment_score(
                    review,
                    tokenizer,
                    model,
                )

                scores.append(score)

                progress = (index + 1) / total_reviews

                progress_bar.progress(progress)

                status.write(
                    f"Analysing review {index + 1:,} of {total_reviews:,}"
                )

            progress_bar.empty()
            status.empty()

            df["sentiment_score"] = scores
            df["overall"] = df["sentiment_score"].apply(map_sentiment)

            st.session_state["analysed_reviews"] = df

            st.success("Sentiment analysis complete.")

    # Persist results after reruns
        if "analysed_reviews" in st.session_state:

            analysed_df = st.session_state["analysed_reviews"]

            st.subheader("Analysis Results")

            sentiment_counts = (
                analysed_df["overall"]
                .value_counts()
            )

            positive = int(sentiment_counts.get("Positive", 0))
            neutral = int(sentiment_counts.get("Neutral", 0))
            negative = int(sentiment_counts.get("Negative", 0))

            c1, c2, c3 = st.columns(3)

            c1.metric("Positive", positive)
            c2.metric("Neutral", neutral)
            c3.metric("Negative", negative)

            st.dataframe(
                analysed_df.head(20),
                use_container_width=True,
            )

            csv_data = analysed_df.to_csv(index=False)

            original_name = uploaded_file.name.rsplit(".", 1)[0]

            output_filename = (
                f"{original_name}_updated_"
                f"{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
            )

            st.download_button(
                label="Download Analysed Reviews",
                data=csv_data,
                file_name=output_filename,
                mime="text/csv",
                use_container_width=True,
            )

            st.info(
                "You can upload this downloaded file to the "
                "Business Recommendations and Data Insights pages."
            )

    except Exception as exc:

        Logger.exception(f"Unexpected application error: {exc}")

        st.error(
            "Something went wrong while processing the file."
        )

        with st.expander("Technical details"):
            st.code(str(exc))

if __name__ == "__main__":
    main()