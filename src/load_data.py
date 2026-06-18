from pathlib import Path
from typing import Optional

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(
    dataset_name: str,
    download_dir: Path,
) -> Path:
    """
    Download and unzip the Kaggle dataset.

    For this project, dataset_name is:
        crowdflower/twitter-airline-sentiment
    """

    download_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = download_dir / "kaggle_airline_sentiment"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    existing_csv = find_tweets_csv_if_exists(extracted_dir)

    if existing_csv is not None:
        print(f"Tweets.csv already exists here: {existing_csv}")
        return existing_csv

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading Kaggle dataset: {dataset_name}")

    api.dataset_download_files(
        dataset=dataset_name,
        path=str(extracted_dir),
        unzip=True,
    )

    tweets_csv_path = find_tweets_csv(extracted_dir)

    return tweets_csv_path


def find_tweets_csv_if_exists(data_dir: Path) -> Optional[Path]:
    """
    Check if Tweets.csv already exists.
    """

    csv_files = list(data_dir.rglob("Tweets.csv"))

    if csv_files:
        return csv_files[0]

    return None


def find_tweets_csv(data_dir: Path) -> Path:
    """
    Find Tweets.csv inside the downloaded Kaggle dataset folder.
    """

    csv_files = list(data_dir.rglob("Tweets.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "Could not find Tweets.csv after downloading the Kaggle dataset. "
            "Check whether the Kaggle download worked correctly."
        )

    return csv_files[0]


def load_airline_tweets_from_kaggle(
    dataset_name: str = "crowdflower/twitter-airline-sentiment",
    raw_output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the Twitter US Airline Sentiment dataset from Kaggle.

    The Kaggle dataset contains many columns, but for this project we need:

        text                -> tweet text
        airline_sentiment   -> sentiment label

    We rename them into:

        text
        label

    so that the rest of the project remains clean.
    """

    raw_dir = raw_output_path.parent if raw_output_path is not None else Path("data/raw")

    tweets_csv_path = download_kaggle_dataset(
        dataset_name=dataset_name,
        download_dir=raw_dir,
    )

    print(f"\nLoading CSV file: {tweets_csv_path}")

    original_df = pd.read_csv(tweets_csv_path)

    print("\nOriginal Kaggle columns:")
    print(list(original_df.columns))

    required_columns = ["text", "airline_sentiment"]

    for column in required_columns:
        if column not in original_df.columns:
            raise ValueError(
                f"Expected column '{column}' was not found. "
                f"Available columns are: {list(original_df.columns)}"
            )

    df = pd.DataFrame({
        "text": original_df["text"].astype(str),
        "label": original_df["airline_sentiment"].astype(str),
    })

    df = df.dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    print("\nNormalized dataset preview:")
    print(df.head())

    print("\nClass distribution:")
    print(df["label"].value_counts())

    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_output_path, index=False)
        print(f"\nSaved normalized Kaggle dataset to: {raw_output_path}")

    return df