"""
build_source_tokens.py

Generates data/processed/source_tokens.csv from the raw project data:
  - Sentence-level/all_languages_sentence_strategy_alignment.csv
  - Word-level/{lang}_word_alignment.csv  (one file per language)
  - Task1/strategy_annotations_template.csv

One row per SOURCE token (long format), with its edit operation and
binary masks. This is the token-level companion to pairs_with_splits.csv,
which stores the same information as compact per-pair JSON lists instead.

Run this from the repo root, e.g.:
    python3 scripts/build_source_tokens.py

Output:
    data/processed/source_tokens.csv
"""

import pandas as pd

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
SENTENCE_LEVEL_DIR = REPO_ROOT / "Sentence-level"
WORD_LEVEL_DIR = REPO_ROOT / "Word-level"
TASK1_DIR = REPO_ROOT / "Task1"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["ar", "ca", "en", "es", "fr", "it"]


def classify_link(row):
    cardinality = row["alignment_cardinality"]
    status = row["alignment_status"]

    if status == "source_only":
        return "DELETE"
    if status == "target_only":
        return "INSERT"
    if cardinality == "one_to_one":
        return "KEEP" if row["candidate_operation"] == "retention" else "SUBSTITUTE"
    if cardinality == "one_source_to_many_targets":
        return "SPLIT"
    if cardinality == "many_sources_to_one_target":
        return "MERGE"
    if cardinality == "many_to_many":
        return "COMPLEX"
    return "UNKNOWN"


# Same collapsing logic as build_pairs_with_splits.py: a single source token
# can appear in multiple link rows (e.g. SPLIT, COMPLEX), so we collapse to
# one label per token using this priority order.
OPERATION_PRIORITY = ["COMPLEX", "SPLIT", "MERGE", "SUBSTITUTE", "DELETE", "INSERT", "KEEP"]

def resolve_token_operation(link_labels):
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

    # 2. Build a crosswalk: sentence_alignment_id -> pair_id, language, collection, document_id
    #    (same join used in build_pairs_with_splits.py)

    crosswalk = sentence_level[
        ["sentence_alignment_id", "language_code", "source_text", "target_text"]
    ].merge(
        template[["pair_id", "language", "collection", "document_id", "source_text", "target_text"]],
        left_on=["language_code", "source_text", "target_text"],
        right_on=["language", "source_text", "target_text"],
        how="inner",
    )

    assert len(crosswalk) == 1930, (
        f"Crosswalk join produced {len(crosswalk)} rows, expected 1930 "
        "(check for duplicate source/target text pairs)"
    )

    id_map = crosswalk.set_index("sentence_alignment_id")["pair_id"].to_dict()
    pair_context = crosswalk.set_index("pair_id")[["language", "collection", "document_id"]]

    # 3. Build the long source-token table, one language at a time

    all_rows = []

    for lang in LANGUAGES:
        wl_path = WORD_LEVEL_DIR / f"{lang}_word_alignment.csv"
        wl = pd.read_csv(wl_path, encoding="utf-8-sig", low_memory=False)
        wl["pair_id"] = wl["sentence_alignment_id"].map(id_map)
        wl["link_label"] = wl.apply(classify_link, axis=1)

        src = wl[wl["source_token_index"].notna()].sort_values(
            ["pair_id", "source_token_index"]
        )

        collapsed = src.groupby(["pair_id", "source_token_index"]).agg(
            source_token=("source_token", "first"),
            source_is_punctuation=("source_is_punctuation", "first"),
            link_labels=("link_label", list),
        )
        collapsed["operation"] = collapsed["link_labels"].apply(resolve_token_operation)
        collapsed = collapsed.drop(columns="link_labels").reset_index()

        all_rows.append(collapsed)

    source_tokens = pd.concat(all_rows, ignore_index=True)
    source_tokens = source_tokens.rename(
        columns={"source_token_index": "token_index", "source_token": "token"}
    )

    # 4. Attach pair-level context (language, collection, document_id)

    source_tokens = source_tokens.merge(
        pair_context, on="pair_id", how="left"
    )

    # 5. Binary masks

    source_tokens["is_punctuation"] = source_tokens["source_is_punctuation"].astype(int)
    source_tokens["is_content"] = 1 - source_tokens["is_punctuation"]

    for op in ["KEEP", "DELETE", "SUBSTITUTE", "SPLIT", "MERGE", "COMPLEX"]:
        source_tokens[f"is_{op.lower()}"] = (source_tokens["operation"] == op).astype(int)

    # content_edit: an edit (non-KEEP) that lands on a content (non-punctuation) token
    source_tokens["content_edit"] = (
        (source_tokens["operation"] != "KEEP") & (source_tokens["is_content"] == 1)
    ).astype(int)

    # 6. Column order

    source_tokens = source_tokens[
        [
            "pair_id",
            "language",
            "collection",
            "document_id",
            "token_index",
            "token",
            "operation",
            "is_punctuation",
            "is_content",
            "is_keep",
            "is_delete",
            "is_substitute",
            "is_split",
            "is_merge",
            "is_complex",
            "content_edit",
        ]
    ]

    # 7. Sanity checks before saving

    op_mask_sum = source_tokens[
        ["is_keep", "is_delete", "is_substitute", "is_split", "is_merge", "is_complex"]
    ].sum(axis=1)
    assert (op_mask_sum == 1).all(), "Every token should have exactly one operation mask set to 1"

    punctuation_content_edit = source_tokens[
        (source_tokens["is_punctuation"] == 1) & (source_tokens["content_edit"] == 1)
    ]
    assert len(punctuation_content_edit) == 0, "content_edit must never be set on a punctuation token"

    print(f"Final source_tokens.csv: {len(source_tokens)} rows, {len(source_tokens.columns)} columns")
    print("Columns:", list(source_tokens.columns))
    print("Operation counts:")
    print(source_tokens["operation"].value_counts())

    # 8. Save

    out_path = OUTPUT_DIR / "source_tokens.csv"
    source_tokens.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()