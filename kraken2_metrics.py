# -*- coding: utf-8 -*-
"""Kraken2 official-baseline metrics (run after Kraken2 has produced .k2.out
files under WSL): parse .k2.out -> taxid -> 25-class predictions (target
species = their own class; anything above species rank or unclassified =
background) -> macro-F1 / accuracy -> results/eval_*_kraken2.json
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(os.environ.get("AQUAPATHID_DATA", "./AquaPathID_data"))
READS = DATA / "reads"
WORK = DATA / "kraken2_work"
DB = DATA / "kraken2_db"
RES = Path(__file__).resolve().parent / "results"
SETS = {"t1_seen_bact": "t1_seen", "t2_unseen_bact": "t2_unseen",
        "t3_reads_bact": "t3_reads", "t4_pond_mix": "t4_pond"}

label_map = json.loads((READS / "label_map.json").read_text(encoding="utf-8"))
targets = label_map["bacteria"]
decoy_id = label_map["bacteria_decoy_id"]
inv_targets = {v: k for k, v in targets.items()}

# taxid -> slug (inverted from the DB's seqid2taxid.map)
taxid2slug = {}
for line in open(DB / "seqid2taxid.map", encoding="utf-8"):
    seqid, taxid = line.split("\t")
    taxid2slug[taxid.strip()] = seqid.split("|", 1)[0]

for npz_name, short in SETS.items():
    z = np.load(READS / f"{npz_name}.npz", allow_pickle=True)
    y_true = [int(v) for v in z["label"]]
    preds, n_classified = [], 0
    for line in open(WORK / f"{npz_name}.k2.out", encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        status, taxid = f[0], f[2]
        if status == "U" or taxid not in taxid2slug:
            preds.append(decoy_id)          # unclassified / above species rank -> background
        else:
            slug = taxid2slug[taxid]
            preds.append(targets.get(slug, decoy_id))
            n_classified += 1
    assert len(preds) == len(y_true), f"{npz_name}: {len(preds)} != {len(y_true)}"

    macro = f1_score(y_true, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_true, preds)
    labels_sorted = sorted(set(y_true) | set(preds))
    p, r, f, s = precision_recall_fscore_support(
        y_true, preds, labels=labels_sorted, zero_division=0)
    per_class = {}
    for j, cid in enumerate(labels_sorted):
        slug = inv_targets.get(cid, "BACKGROUND")
        per_class[slug] = {"precision": round(float(p[j]), 4),
                           "recall": round(float(r[j]), 4),
                           "f1": round(float(f[j]), 4),
                           "support": int(s[j])}
    out = {"acc": round(float(acc), 4), "macro_f1": round(float(macro), 4),
           "n_reads": len(y_true), "n_classified": n_classified,
           "per_class": per_class}
    out_path = RES / f"eval_{short}_kraken2.json"
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{short}: mF1={macro:.4f} acc={acc:.4f} classified="
          f"{n_classified}/{len(y_true)} -> {out_path.name}")
print("KRAKEN2_METRICS_DONE")
