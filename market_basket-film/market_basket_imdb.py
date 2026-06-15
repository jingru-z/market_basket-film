from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Kaggle dataset identifier from the assignment.
DATASET_ID = "harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows"

# Keep these placeholders in the final public version of the project.
# For local execution, prefer setting KAGGLE_USERNAME and KAGGLE_KEY as
# environment variables instead of writing secrets into this file.
KAGGLE_USERNAME = "xxxxxx"
KAGGLE_KEY = "xxxxxx"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

STAR_COLUMNS = ["Star1", "Star2", "Star3", "Star4"]


# If MIN_SUPPORT is in (0, 1], it is treated as a fraction of all baskets.
# If it is greater than 1, it is treated as an absolute support count.
DEFAULT_MIN_SUPPORT = 0.003
DEFAULT_MIN_CONFIDENCE = 0.50

# Threshold settings used to compare how support/confidence affect the results.
THRESHOLD_EXPERIMENTS = [
    ("High support", 0.020, 0.50),
    ("Loose", 0.002, 0.20),
    ("Final", DEFAULT_MIN_SUPPORT, DEFAULT_MIN_CONFIDENCE),
]

# The IMDb task has four actor columns, so size 4 is the natural maximum.
DEFAULT_MAX_ITEMSET_SIZE = 4

# Global switch requested in the assignment: use a subsample or all rows.
USE_SUBSAMPLE = False
SUBSAMPLE_SIZE = 200
RANDOM_STATE = 42


Itemset = Tuple[str, ...]
Transaction = Set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market-basket analysis on IMDb movie actors using Apriori."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not call Kaggle; use an existing CSV under data/raw.",
    )
    parser.add_argument(
        "--use-subsample",
        action="store_true",
        default=USE_SUBSAMPLE,
        help="Analyze a reproducible subsample instead of the whole dataset.",
    )
    parser.add_argument(
        "--subsample-size",
        type=int,
        default=SUBSAMPLE_SIZE,
        help="Number of movies to keep when --use-subsample is enabled.",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=DEFAULT_MIN_SUPPORT,
        help=(
            "Minimum support. Use a fraction such as 0.02, or an absolute "
            "count such as 10."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum confidence for association rules.",
    )
    parser.add_argument(
        "--max-itemset-size",
        type=int,
        default=DEFAULT_MAX_ITEMSET_SIZE,
        help="Maximum itemset size considered by Apriori.",
    )
    parser.add_argument(
        "--compare-thresholds",
        action="store_true",
        help="Run additional threshold comparison experiments and save a CSV summary.",
    )
    return parser.parse_args()


def configure_kaggle_credentials() -> None:
    """Set Kaggle credentials from constants only when real values are given."""
    if KAGGLE_USERNAME != "xxxxxx":
        os.environ.setdefault("KAGGLE_USERNAME", KAGGLE_USERNAME)
    if KAGGLE_KEY != "xxxxxx":
        os.environ.setdefault("KAGGLE_KEY", KAGGLE_KEY)


def has_kaggle_credentials() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def download_dataset() -> None:
    """Download and unzip the dataset through the Kaggle command-line API."""
    configure_kaggle_credentials()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials were not found. Set KAGGLE_USERNAME and "
            "KAGGLE_KEY in your terminal, or temporarily replace the xxxxxx "
            "placeholders in the script while running locally."
        )

    kaggle_executable = shutil.which("kaggle")
    if kaggle_executable:
        command = [kaggle_executable]
    else:
        command = [sys.executable, "-m", "kaggle"]

    command += [
        "datasets",
        "download",
        "-d",
        DATASET_ID,
        "-p",
        str(RAW_DATA_DIR),
        "--unzip",
    ]

    print("Downloading dataset from Kaggle...")
    subprocess.run(command, check=True)
    print(f"Dataset downloaded under: {RAW_DATA_DIR.resolve()}")


def find_dataset_csv() -> Path:
    """Find the CSV file containing the four actor columns required by the task."""
    csv_files = sorted(RAW_DATA_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found under {RAW_DATA_DIR.resolve()}. "
            "Run without --skip-download or manually place the Kaggle CSV there."
        )

    for csv_path in csv_files:
        try:
            columns = pd.read_csv(csv_path, nrows=0).columns
        except Exception:
            continue
        if all(column in columns for column in STAR_COLUMNS):
            return csv_path

    raise FileNotFoundError(
        "A CSV file was found, but none contained all required columns: "
        + ", ".join(STAR_COLUMNS)
    )


def normalize_actor_name(value: object) -> str | None:
    """Clean and standardize actor names before using them as basket items."""
    if pd.isna(value):
        return None

    name = unicodedata.normalize("NFKC", str(value))
    name = name.replace("\u00a0", " ")
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" ,;")

    if not name:
        return None

    return name


def row_to_transaction(row: pd.Series) -> Transaction:
    """Convert one movie row into a basket of unique standardized actor names."""
    actors: Transaction = set()
    for column in STAR_COLUMNS:
        actor = normalize_actor_name(row.get(column))
        if actor:
            actors.add(actor)
    return actors


def load_transactions(
    csv_path: Path,
    use_subsample: bool,
    subsample_size: int,
) -> List[Transaction]:
    """Load movie baskets from the four Star columns."""
    df = pd.read_csv(csv_path, usecols=STAR_COLUMNS, dtype=str)

    if use_subsample:
        if subsample_size <= 0:
            raise ValueError("--subsample-size must be positive.")
        sample_size = min(subsample_size, len(df))
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    transactions = [row_to_transaction(row) for _, row in df.iterrows()]
    transactions = [transaction for transaction in transactions if transaction]

    if not transactions:
        raise ValueError("No valid actor baskets were created from the dataset.")

    return transactions


def minimum_support_count(min_support: float, n_transactions: int) -> int:
    """Convert a fractional or absolute support threshold into a count."""
    if min_support <= 0:
        raise ValueError("Minimum support must be positive.")
    if min_support <= 1:
        return max(1, math.ceil(min_support * n_transactions))
    return int(math.ceil(min_support))


def count_singletons(transactions: Sequence[Transaction]) -> Counter[Itemset]:
    counts: Counter[Itemset] = Counter()
    for transaction in transactions:
        for item in transaction:
            counts[(item,)] += 1
    return counts


def filter_frequent(
    counts: Counter[Itemset],
    min_support_count_value: int,
) -> Dict[Itemset, int]:
    return {
        itemset: count
        for itemset, count in counts.items()
        if count >= min_support_count_value
    }


def generate_candidates(previous_frequents: Iterable[Itemset], k: int) -> Set[Itemset]:
    """Generate Apriori candidates of size k from frequent itemsets of size k - 1."""
    previous_set = set(previous_frequents)
    previous_list = sorted(previous_set)
    candidates: Set[Itemset] = set()

    for left_index in range(len(previous_list)):
        for right_index in range(left_index + 1, len(previous_list)):
            union = tuple(sorted(set(previous_list[left_index]) | set(previous_list[right_index])))
            if len(union) != k:
                continue

            all_subsets_are_frequent = all(
                tuple(sorted(subset)) in previous_set
                for subset in itertools.combinations(union, k - 1)
            )
            if all_subsets_are_frequent:
                candidates.add(union)

    return candidates


def count_candidates(
    transactions: Sequence[Transaction],
    candidates: Set[Itemset],
    k: int,
) -> Counter[Itemset]:
    """Count candidate itemsets by scanning each basket once."""
    counts: Counter[Itemset] = Counter()
    if not candidates:
        return counts

    for transaction in transactions:
        if len(transaction) < k:
            continue
        for combination in itertools.combinations(sorted(transaction), k):
            if combination in candidates:
                counts[combination] += 1

    return counts


def apriori(
    transactions: Sequence[Transaction],
    min_support: float,
    max_itemset_size: int,
) -> Tuple[Dict[int, Dict[Itemset, int]], Dict[Itemset, int], int]:
    """Run the Apriori algorithm and return frequent itemsets by size."""
    if max_itemset_size < 1:
        raise ValueError("max_itemset_size must be at least 1.")

    n_transactions = len(transactions)
    min_count = minimum_support_count(min_support, n_transactions)

    frequent_by_size: Dict[int, Dict[Itemset, int]] = {}
    all_support_counts: Dict[Itemset, int] = {}

    singleton_counts = count_singletons(transactions)
    frequent_1 = filter_frequent(singleton_counts, min_count)
    frequent_by_size[1] = frequent_1
    all_support_counts.update(frequent_1)

    previous_frequents = set(frequent_1)

    for k in range(2, max_itemset_size + 1):
        candidates = generate_candidates(previous_frequents, k)
        candidate_counts = count_candidates(transactions, candidates, k)
        frequent_k = filter_frequent(candidate_counts, min_count)

        if not frequent_k:
            break

        frequent_by_size[k] = frequent_k
        all_support_counts.update(frequent_k)
        previous_frequents = set(frequent_k)

    return frequent_by_size, all_support_counts, min_count


def support_fraction(count: int, n_transactions: int) -> float:
    return count / n_transactions


def generate_association_rules(
    all_support_counts: Dict[Itemset, int],
    n_transactions: int,
    min_confidence: float,
) -> pd.DataFrame:
    """Generate rules A -> B from frequent itemsets."""
    rules = []

    for itemset, itemset_count in all_support_counts.items():
        if len(itemset) < 2:
            continue

        itemset_items = set(itemset)
        for antecedent_size in range(1, len(itemset)):
            for antecedent_tuple in itertools.combinations(itemset, antecedent_size):
                antecedent = tuple(sorted(antecedent_tuple))
                consequent = tuple(sorted(itemset_items - set(antecedent)))

                antecedent_count = all_support_counts.get(antecedent)
                consequent_count = all_support_counts.get(consequent)
                if not antecedent_count or not consequent_count:
                    continue

                confidence = itemset_count / antecedent_count
                if confidence < min_confidence:
                    continue

                itemset_support = support_fraction(itemset_count, n_transactions)
                consequent_support = support_fraction(consequent_count, n_transactions)
                lift = confidence / consequent_support if consequent_support else float("inf")

                rules.append(
                    {
                        "antecedent": " | ".join(antecedent),
                        "consequent": " | ".join(consequent),
                        "support_count": itemset_count,
                        "support": itemset_support,
                        "confidence": confidence,
                        "lift": lift,
                    }
                )

    rules_df = pd.DataFrame(rules)
    if rules_df.empty:
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "support_count",
                "support",
                "confidence",
                "lift",
            ]
        )

    return rules_df.sort_values(
        by=["confidence", "lift", "support_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def frequent_itemsets_to_dataframe(
    frequent_by_size: Dict[int, Dict[Itemset, int]],
    n_transactions: int,
) -> pd.DataFrame:
    rows = []
    for size, itemsets in frequent_by_size.items():
        for itemset, count in itemsets.items():
            rows.append(
                {
                    "size": size,
                    "itemset": " | ".join(itemset),
                    "support_count": count,
                    "support": support_fraction(count, n_transactions),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["size", "itemset", "support_count", "support"])

    return result.sort_values(
        by=["size", "support_count", "itemset"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def save_outputs(frequent_df: pd.DataFrame, rules_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frequent_path = OUTPUT_DIR / "frequent_itemsets.csv"
    rules_path = OUTPUT_DIR / "association_rules.csv"

    frequent_df.to_csv(frequent_path, index=False)
    rules_df.to_csv(rules_path, index=False)

    print(f"Saved frequent itemsets to: {frequent_path.resolve()}")
    print(f"Saved association rules to: {rules_path.resolve()}")


def run_threshold_comparison(
    transactions: Sequence[Transaction],
    max_itemset_size: int,
) -> pd.DataFrame:
    """Compare selected support/confidence settings for the report."""
    rows = []
    n_transactions = len(transactions)

    for setting_name, min_support, min_confidence in THRESHOLD_EXPERIMENTS:
        frequent_by_size, all_support_counts, min_count = apriori(
            transactions=transactions,
            min_support=min_support,
            max_itemset_size=max_itemset_size,
        )
        rules_df = generate_association_rules(
            all_support_counts=all_support_counts,
            n_transactions=n_transactions,
            min_confidence=min_confidence,
        )

        size_counts = {
            f"frequent_size_{size}": len(itemsets)
            for size, itemsets in frequent_by_size.items()
        }
        row = {
            "setting": setting_name,
            "min_support": min_support,
            "min_confidence": min_confidence,
            "min_support_count": min_count,
            "frequent_itemsets": sum(len(itemsets) for itemsets in frequent_by_size.values()),
            "association_rules": len(rules_df),
        }
        row.update(size_counts)
        rows.append(row)

    comparison_df = pd.DataFrame(rows).fillna(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_DIR / "threshold_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print(f"Saved threshold comparison to: {comparison_path.resolve()}")
    print("\n=== Threshold comparison ===")
    print(comparison_df.to_string(index=False))

    return comparison_df


def print_summary(
    transactions: Sequence[Transaction],
    min_support_count_value: int,
    frequent_df: pd.DataFrame,
    rules_df: pd.DataFrame,
) -> None:
    unique_actors = sorted({actor for transaction in transactions for actor in transaction})
    basket_sizes = [len(transaction) for transaction in transactions]

    print("\n=== Dataset summary ===")
    print(f"Movies/baskets analyzed: {len(transactions)}")
    print(f"Unique actors/items: {len(unique_actors)}")
    print(f"Average basket size: {sum(basket_sizes) / len(basket_sizes):.2f}")
    print(f"Minimum support count: {min_support_count_value}")

    print("\n=== Top frequent itemsets ===")
    if frequent_df.empty:
        print("No frequent itemsets found with the current threshold.")
    else:
        print(frequent_df.head(20).to_string(index=False))

    print("\n=== Top association rules ===")
    if rules_df.empty:
        print("No association rules found with the current confidence threshold.")
    else:
        print(rules_df.head(20).to_string(index=False))


def main() -> None:
    args = parse_args()

    if not args.skip_download:
        existing_csvs = list(RAW_DATA_DIR.rglob("*.csv")) if RAW_DATA_DIR.exists() else []
        if existing_csvs:
            print("Existing CSV detected under data/raw; skipping download.")
        else:
            download_dataset()

    csv_path = find_dataset_csv()
    print(f"Using dataset file: {csv_path.resolve()}")

    transactions = load_transactions(
        csv_path=csv_path,
        use_subsample=args.use_subsample,
        subsample_size=args.subsample_size,
    )

    if args.compare_thresholds:
        run_threshold_comparison(
            transactions=transactions,
            max_itemset_size=args.max_itemset_size,
        )

    frequent_by_size, all_support_counts, min_count = apriori(
        transactions=transactions,
        min_support=args.min_support,
        max_itemset_size=args.max_itemset_size,
    )

    frequent_df = frequent_itemsets_to_dataframe(
        frequent_by_size=frequent_by_size,
        n_transactions=len(transactions),
    )
    rules_df = generate_association_rules(
        all_support_counts=all_support_counts,
        n_transactions=len(transactions),
        min_confidence=args.min_confidence,
    )

    save_outputs(frequent_df, rules_df)
    print_summary(transactions, min_count, frequent_df, rules_df)


if __name__ == "__main__":
    main()
