"""
build_pairs_with_splits.py

Generates data/processed/pairs_with_splits.csv from the raw project data:
  - Sentence-level/all_languages_sentence_strategy_alignment.csv
  - Word-level/{lang}_word_alignment.csv  (one file per language)
  - Task1/strategy_annotations_template.csv

Run this from the repo root, e.g.:
    python3 scripts/build_pairs_with_splits.py

Output:
    data/processed/pairs_with_splits.csv
"""

import json
from pathlib import Path

import pandas as pd

# 0. Paths

REPO_ROOT = Path(__file__).resolve().parent.parent
SENTENCE_LEVEL_DIR = REPO_ROOT / "Sentence-level"
WORD_LEVEL_DIR = REPO_ROOT / "Word-level"
TASK1_DIR = REPO_ROOT / "Task1"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["ar", "ca", "en", "es", "fr", "it"]

# Maps raw alignment_cardinality + alignment_status into the operation
# vocabulary the notebook expects: KEEP, DELETE, INSERT, SUBSTITUTE, SPLIT,
# MERGE, COMPLEX

def classify_link(row):
    cardinality = row["alignment_cardinality"]
    status = row["alignment_status"]

    if status == "source_only":
        return "DELETE"
    if status == "target_only":
        return "INSERT"
    # status == "aligned" from here on
    if cardinality == "one_to_one":
        return "KEEP" if row["candidate_operation"] == "retention" else "SUBSTITUTE"
    if cardinality == "one_source_to_many_targets":
        return "SPLIT"
    if cardinality == "many_sources_to_one_target":
        return "MERGE"
    if cardinality == "many_to_many":
        return "COMPLEX"
    return "UNKNOWN"


# A single source (or target) token can appear in MULTIPLE link rows, e.g.
# when one source token aligns to several target tokens ("SPLIT") or several
# source tokens collapse onto one target token ("MERGE"). We collapse those
# duplicate rows into one label per token using this priority order, since
# "COMPLEX"/"SPLIT"/"MERGE" are more informative than a plain KEEP/SUBSTITUTE
# link that happens to co-occur on the same token.

OPERATION_PRIORITY = ["COMPLEX", "SPLIT", "MERGE", "SUBSTITUTE", "DELETE", "INSERT", "KEEP"]

def resolve_token_operation(link_labels):
    """Given all link-level labels seen for one token, pick the single most
    informative operation label, using OPERATION_PRIORITY."""
    label_set = set(link_labels)
    for candidate in OPERATION_PRIORITY:
        if candidate in label_set:
            return candidate
    return link_labels[0]

def main():
    # 1. Load the two "table" sources
    sentence_level = pd.read_csv(
        SENTENCE_LEVEL_DIR / "all_languages_sentence_strategy_alignment.csv",
        encoding="utf-8-sig",
    )
    template = pd.read_csv(
        TASK1_DIR / "strategy_annotations_template.csv",
        encoding="utf-8-sig",
    )

    assert len(template) == 1930, f"Expected 1930 template rows, got {len(template)}"

    # 2. Build a crosswalk: sentence_alignment_id -> pair_id
    #    (these two files use different ID schemes, so we join on the
    #    text itself, which is guaranteed unique per pair)

    crosswalk = sentence_level[
        ["sentence_alignment_id", "language_code", "document_id", "source_text", "target_text"]
    ].merge(
        template[["pair_id", "language", "source_text", "target_text"]],
        left_on=["language_code", "source_text", "target_text"],
        right_on=["language", "source_text", "target_text"],
        how="inner",
    )

    assert len(crosswalk) == 1930, (
        f"Crosswalk join produced {len(crosswalk)} rows, expected 1930 "
        "(check for duplicate source/target text pairs)"
    )

    id_map = crosswalk.set_index("sentence_alignment_id")["pair_id"].to_dict()

    # 3. Base pair table: template columns + sentence-level numeric columns
    sentence_level_numeric = sentence_level.merge(
        crosswalk[["sentence_alignment_id", "pair_id"]],
        on="sentence_alignment_id",
        how="inner",
    )[
        [
            "pair_id",
            "source_token_count",
            "target_token_count",
            "aligned_link_count",
            "source_coverage_pct",
            "target_coverage_pct",
        ]
    ].rename(
        columns={
            "source_token_count": "n_source_tokens",
            "target_token_count": "n_target_tokens",
            "aligned_link_count": "n_alignment_links",
            "source_coverage_pct": "source_alignment_coverage",
            "target_coverage_pct": "target_alignment_coverage",
        }
    )

    pairs = template.merge(sentence_level_numeric, on="pair_id", how="left")

    # The template's own `document_id` column is actually just the raw corpus
    # filename (identical for every pair in a collection) rather than a real
    # per-document identifier. The sentence-level file's `document_id` is the
    # correct, finer-grained one (e.g. real multi-sentence documents for the
    # iDEM collections), so we overwrite it here.
    
    document_id_map = crosswalk.set_index("pair_id")["document_id"].to_dict()
    pairs["document_id"] = pairs["pair_id"].map(document_id_map)

    # 4. Token / operation lists, derived from the word-level files
    source_tokens_by_pair = {}
    source_ops_by_pair = {}
    target_tokens_by_pair = {}
    target_ops_by_pair = {}

    for lang in LANGUAGES:
        wl_path = WORD_LEVEL_DIR / f"{lang}_word_alignment.csv"
        wl = pd.read_csv(wl_path, encoding="utf-8-sig", low_memory=False)
        wl["pair_id"] = wl["sentence_alignment_id"].map(id_map)
        wl["link_label"] = wl.apply(classify_link, axis=1)

        # source side: collapse to one row per (pair_id, source_token_index)
        src = wl[wl["source_token_index"].notna()].sort_values(
            ["pair_id", "source_token_index"]
        )
        for pid, group in src.groupby("pair_id"):
            collapsed = (
                group.groupby("source_token_index")
                .agg(
                    source_token=("source_token", "first"),
                    link_labels=("link_label", list),
                )
                .sort_index()
            )
            source_tokens_by_pair[pid] = collapsed["source_token"].tolist()
            source_ops_by_pair[pid] = [
                resolve_token_operation(labels) for labels in collapsed["link_labels"]
            ]

        # target side: collapse to one row per (pair_id, target_token_index)
        tgt = wl[wl["target_token_index"].notna()].sort_values(
            ["pair_id", "target_token_index"]
        )
        for pid, group in tgt.groupby("pair_id"):
            collapsed = (
                group.groupby("target_token_index")
                .agg(
                    target_token=("target_token", "first"),
                    link_labels=("link_label", list),
                )
                .sort_index()
            )
            target_tokens_by_pair[pid] = collapsed["target_token"].tolist()
            target_ops_by_pair[pid] = [
                resolve_token_operation(labels) for labels in collapsed["link_labels"]
            ]

    pairs["source_tokens"] = pairs["pair_id"].map(source_tokens_by_pair).apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else None
    )
    pairs["source_operations"] = pairs["pair_id"].map(source_ops_by_pair).apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else None
    )
    pairs["target_tokens"] = pairs["pair_id"].map(target_tokens_by_pair).apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else None
    )
    pairs["target_operations"] = pairs["pair_id"].map(target_ops_by_pair).apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else None
    )

    # 5. Sanity checks before saving
    n_missing_tokens = pairs["source_tokens"].isna().sum()
    if n_missing_tokens:
        print(
            f"WARNING: {n_missing_tokens} pairs have no source_tokens list "
            "(no matching rows found in the word-level files for that pair_id)."
        )

    print(f"Final pairs_with_splits.csv: {len(pairs)} rows, {len(pairs.columns)} columns")
    print("Columns:", list(pairs.columns))

    # 6. Save
    out_path = OUTPUT_DIR / "pairs_with_splits.csv"
    pairs.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()