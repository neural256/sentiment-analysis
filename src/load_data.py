from pathlib import Path
from typing import Optional

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(
    dataset_name: str,
    download_dir: Path,
) -> Path:

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
    raw_dir: Optional[Path] = None,
    processed_output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the Twitter US Airline Sentiment dataset from Kaggle.

    The Kaggle dataset contains many columns, but for this project we need:

        text                -> tweet text
        airline_sentiment   -> sentiment label

    We rename them into:

        text
        label

    Data layout:
        Raw download  -> data/raw/        (untouched Kaggle files)
        Cleaned CSV   -> data/processed/  (this function's output, reused on later runs)
    """

    if raw_dir is None:
        raw_dir = Path("data/raw")

    # If we already cleaned this dataset on a previous run, reuse the processed
    # file instead of downloading and cleaning the raw data again.
    if processed_output_path is not None and processed_output_path.exists():
        print(f"Loading cleaned dataset from: {processed_output_path}")
        return pd.read_csv(processed_output_path)

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

    if processed_output_path is not None:
        processed_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_output_path, index=False)
        print(f"\nSaved cleaned dataset to: {processed_output_path}")

    return df


def load_airline_tweets_from_openml(arff_path) -> pd.DataFrame:
    """
    Load the OpenML 'Airlines-Tweets-Sentiments' dataset (ARFF format).

    File columns: _id, tweet_text, tweet_lang, tweet_sentiment_value (0=neg, 1=neu, 2=pos).
    We keep tweet_text -> text and map the numeric sentiment -> label, so the
    output matches the Kaggle loader (columns: text, label).
    """
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    lines = Path(arff_path).read_text(encoding="utf-8").splitlines()
    data_start = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("@data"))

    texts, labels = [], []
    for line in lines[data_start + 1:]:
        line = line.strip()
        if not line:
            continue
        try:
            _id, rest = line.split(",", 1)                 # peel off the id
            text_part, _lang, value = rest.rsplit(",", 2)  # peel lang + value off the right
            value = int(value.strip())
            if value not in label_map:
                continue
            text = text_part.strip()
            if text.startswith("'") and text.endswith("'"):
                text = text[1:-1]                          # drop the surrounding quotes
            text = text.replace("\\'", "'")                # unescape \' -> '
        except ValueError:
            continue                                       # skip any malformed row
        texts.append(text)
        labels.append(label_map[value])

    df = pd.DataFrame({"text": texts, "label": labels})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    print(f"\nLoaded OpenML dataset: {len(df)} rows from {arff_path}")
    print(df.head())
    print("\nClass distribution:")
    print(df["label"].value_counts())

    return df
