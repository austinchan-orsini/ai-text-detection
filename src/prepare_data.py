"""Build the checkpoint dataset from M4 (Wang et al., EACL 2024).

Reads the English M4 jsonl files (arxiv, reddit, wikihow, wikipedia) for the
five generators that cover all four domains, deduplicates human texts, assigns
every text to a "prompt group" (the human text it was paired with), and splits
70/15/15 BY GROUP so that a human passage and all machine versions of it land
in the same split (no topic leakage).

Usage:
    git clone https://github.com/mbzuai-nlp/M4.git   # data/ folder holds the jsonl files
    python src/prepare_data.py --m4_dir M4/data --out data/m4_subset.parquet
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ["arxiv", "reddit", "wikihow", "wikipedia"]
# file-name spelling differs across domains in M4.
# (bloomz is excluded: its M4 files use a different schema)
GENERATORS = {
    "chatgpt": ["chatGPT", "chatgpt"],
    "davinci": ["davinci"],
    "cohere": ["cohere"],
    "dolly": ["dolly", "dolly2"],
}
MIN_CHARS = 200      # drop empty / degenerate generations
MAX_CHARS = 2000     # truncate long passages (keeps runs fast, similar lengths)


def clean(t):
    if not isinstance(t, str):
        return None
    t = " ".join(t.split())
    return t[:MAX_CHARS] if len(t) >= MIN_CHARS else None


def gid(text):
    return hashlib.md5(" ".join(str(text).split()).encode()).hexdigest()


def find_file(m4_dir, domain, names):
    for n in names:
        p = Path(m4_dir) / f"{domain}_{n}.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(f"{domain} {names}")


def main(args):
    rows = []
    for d in DOMAINS:
        for g, names in GENERATORS.items():
            with open(find_file(args.m4_dir, d, names)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    h_raw = r.get("human_text")
                    if isinstance(h_raw, list) or isinstance(r.get("machine_text"), list):
                        continue
                    h, m = clean(h_raw), clean(r.get("machine_text"))
                    if h is None:
                        continue
                    group = d + "_" + gid(h_raw)
                    rows.append(dict(text=h, label=0, generator="human", domain=d, group=group))
                    if m is not None and m != h:
                        rows.append(dict(text=m, label=1, generator=g, domain=d, group=group))
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text", "label", "generator"])  # human texts repeat across files
    df = df.drop_duplicates(subset=["text"], keep=False)             # drop any text with both labels

    # grouped 70/15/15 split
    rng = np.random.RandomState(args.seed)
    groups = np.array(sorted(df["group"].unique()))
    rng.shuffle(groups)
    n = len(groups)
    split_of = {}
    for i, g in enumerate(groups):
        split_of[g] = "train" if i < 0.70 * n else ("val" if i < 0.85 * n else "test")
    df["split"] = df["group"].map(split_of)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"saved {len(df)} rows to {args.out}")
    print(pd.crosstab(df["generator"], df["split"], margins=True))
    print(pd.crosstab(df["domain"], df["label"]))
    # sanity: no group crosses splits
    assert df.groupby("group")["split"].nunique().max() == 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m4_dir", default="M4/data")
    ap.add_argument("--out", default="data/m4_subset.parquet")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
