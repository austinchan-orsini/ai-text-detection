"""TF-IDF + logistic regression baseline.

Two settings, evaluated on the VALIDATION split (test split stays untouched):
  seen   : train on human + all 4 generators
  unseen : leave-one-generator-out -- drop generator g from training, evaluate on g
For every generator g the eval set is identical in both settings
(val human texts + val texts from g), so  gap = AUROC_seen(g) - AUROC_unseen(g).
Training sets are balanced 50/50 human/machine and have the same size in both settings.

Usage:
    python src/baseline.py --data data/m4_subset.parquet --out results/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from metrics import all_metrics

GENS = ["chatgpt", "davinci", "cohere", "dolly"]


def balanced_train(df, gens, seed):
    """All train humans + an equal number of machine texts, spread evenly over gens."""
    hum = df[(df.split == "train") & (df.label == 0)]
    per_gen = len(hum) // len(gens)
    parts = [hum]
    for g in gens:
        pool = df[(df.split == "train") & (df.generator == g)]
        parts.append(pool.sample(n=min(per_gen, len(pool)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


class TfidfLR:
    def __init__(self, C=1.0, seed=42):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                                    sublinear_tf=True, lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                    max_features=100_000, sublinear_tf=True)
        self.clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=seed)

    def _x(self, texts, fit=False):
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts, y):
        self.clf.fit(self._x(texts, fit=True), y)
        return self

    def score(self, texts):
        return self.clf.predict_proba(self._x(texts))[:, 1]

    def top_features(self, k=15):
        names = np.concatenate([["w:" + n for n in self.word.get_feature_names_out()],
                                ["c:" + n for n in self.char.get_feature_names_out()]])
        w = self.clf.coef_[0]
        o = np.argsort(w)
        return {"machine": [(names[i], round(float(w[i]), 2)) for i in o[::-1][:k]],
                "human": [(names[i], round(float(w[i]), 2)) for i in o[:k]]}


def eval_set(df, g):
    return df[(df.split == "val") & ((df.label == 0) | (df.generator == g))]


def main(args):
    df = pd.read_parquet(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val = df[df.split == "val"]

    # 1) choose C on validation (seen setting, all val data)
    tr = balanced_train(df, GENS, args.seed)
    c_scores = {}
    for C in [0.3, 1.0, 3.0, 10.0]:
        m = TfidfLR(C, args.seed).fit(tr.text.tolist(), tr.label.values)
        c_scores[C] = all_metrics(val.label.values, m.score(val.text.tolist()))["auroc"]
        print(f"C={C}: val AUROC={c_scores[C]:.4f}")
    C = max(c_scores, key=c_scores.get)
    print("chosen C =", C)

    # 2) seen-generator model
    seen = TfidfLR(C, args.seed).fit(tr.text.tolist(), tr.label.values)
    val = val.assign(score_seen=seen.score(val.text.tolist()))

    rows, per_domain, examples = [], [], {}
    for g in GENS:
        others = [x for x in GENS if x != g]
        tr_g = balanced_train(df, others, args.seed)
        unseen = TfidfLR(C, args.seed).fit(tr_g.text.tolist(), tr_g.label.values)
        ev = eval_set(val, g).copy()
        ev["score_unseen"] = unseen.score(ev.text.tolist())
        ms = all_metrics(ev.label.values, ev.score_seen.values)
        mu = all_metrics(ev.label.values, ev.score_unseen.values)
        rows.append(dict(generator=g, n_eval=len(ev), n_train_seen=len(tr), n_train_unseen=len(tr_g),
                         **{f"seen_{k}": v for k, v in ms.items()},
                         **{f"unseen_{k}": v for k, v in mu.items()},
                         gap_auroc=ms["auroc"] - mu["auroc"]))
        for d, sub in ev.groupby("domain"):
            per_domain.append(dict(generator=g, domain=d,
                                   seen_auroc=all_metrics(sub.label.values, sub.score_seen.values)["auroc"],
                                   unseen_auroc=all_metrics(sub.label.values, sub.score_unseen.values)["auroc"]))
        # missed machine texts (most confidently called human) under the unseen model
        miss = ev[ev.label == 1].nsmallest(3, "score_unseen")
        examples[g] = [dict(domain=r.domain, score=round(float(r.score_unseen), 3), text=r.text[:300])
                       for r in miss.itertuples()]
        print(f"{g}: seen AUROC {ms['auroc']:.4f} | unseen AUROC {mu['auroc']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "baseline_results.csv", index=False)
    pd.DataFrame(per_domain).to_csv(out / "baseline_per_domain.csv", index=False)
    with open(out / "baseline_extras.json", "w") as f:
        json.dump(dict(C_search=c_scores, chosen_C=C, top_features=seen.top_features(),
                       hardest_unseen_examples=examples,
                       overall_seen_val=all_metrics(val.label.values, val.score_seen.values)),
                  f, indent=2)
    print(res.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/m4_subset.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
