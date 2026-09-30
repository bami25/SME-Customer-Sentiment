import chardet
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def ingest_csv(uploaded_file):
    raw_data = uploaded_file.read()
    encoding = chardet.detect(raw_data).get("encoding") or "utf-8"

    uploaded_file.seek(0)

    df = pd.read_csv(uploaded_file, encoding=encoding)

    return df


def standardise_columns(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def validate_schema(df):
    required_columns = ["product_name", "review"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )


def transform_data(df):
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            dayfirst=True
        )

        if "month" not in df.columns:
            df["month"] = df["date"].dt.month_name()

        if "year" not in df.columns:
            df["year"] = df["date"].dt.year

    return df


def predict_sentiment(review, tokenizer, model):
    if not isinstance(review, str) or not review.strip():
        return 0

    encoded = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output = model(**encoded)

    return int(torch.argmax(output.logits, dim=1).item() + 1)


def map_sentiment(score):
    if score in [4, 5]:
        return "Positive"
    elif score == 3:
        return "Neutral"
    elif score in [1, 2]:
        return "Negative"
    return "Unknown"


def run_pipeline(uploaded_file, tokenizer, model):
    df = ingest_csv(uploaded_file)

    df = standardise_columns(df)

    validate_schema(df)

    df = transform_data(df)

    df["sentiment_score"] = df["review"].apply(
        lambda review: predict_sentiment(
            review,
            tokenizer,
            model
        )
    )

    df["overall"] = df["sentiment_score"].apply(
        map_sentiment
    )

    return df