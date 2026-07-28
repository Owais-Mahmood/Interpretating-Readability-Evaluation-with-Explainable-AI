from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from xai_pipeline.contracts import Example
from xai_pipeline.registry import DATASETS

LABELS = [
    "Compression",
    "Explanation",
    "Illocutionary Change",
    "Modulation",
    "Omission",
    "Synonymy",
    "Syntactic Change",
]


@DATASETS.register("simplification_test_set")
class SimplificationDatasetAdapter:
    """Loads Task 2's test_set_with_spans.csv (test_set.csv enriched with
    real deletion/insertion spans from Task 1) into the framework's Example
    contract. Applies the Transposition -> Syntactic Change merge (per
    Nouran's instruction) so every pair has an evaluable official-taxonomy
    label.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._df: pd.DataFrame | None = None

    def _load_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self.path)
            # Merge Transposition into Syntactic Change (agreed with Nouran)
            df["Syntactic Change"] = (
                (df["Syntactic Change"] == 1) | (df["Transposition"] == 1)
            ).astype(int)
            self._df = df
        return self._df

    def load(self, split: str = "test") -> list[Example]:
        df = self._load_dataframe()
        has_spans = "deletion_spans" in df.columns and "insertion_spans" in df.columns

        examples = []
        for _, row in df.iterrows():
            references = {}
            if has_spans:
                references["deletion_spans"] = json.loads(row["deletion_spans"])
                references["insertion_spans"] = json.loads(row["insertion_spans"])

            examples.append(
                Example(
                    example_id=str(row["pair_id"]),
                    inputs={
                        "source_text": row["source_text"],
                        "simplified_text": row["simplified_text"],
                    },
                    labels={label: int(row[label]) for label in LABELS},
                    metadata={
                        "language": row["language"],
                        "group_id": row["group_id"],
                    },
                    references=references,
                )
            )
        return examples

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.path.exists():
            errors.append(f"Dataset file does not exist: {self.path}")
            return errors

        df = self._load_dataframe()

        required_cols = {"pair_id", "language", "source_text", "simplified_text", *LABELS}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {sorted(missing_cols)}")

        if df["pair_id"].duplicated().any():
            errors.append("Duplicate pair_id values found.")

        no_official_label = df[df[LABELS].sum(axis=1) == 0]
        if len(no_official_label) > 0:
            errors.append(
                f"{len(no_official_label)} pairs have zero official-taxonomy labels "
                f"even after the Transposition merge: {no_official_label['pair_id'].tolist()}"
            )

        return errors

    def fingerprint(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]