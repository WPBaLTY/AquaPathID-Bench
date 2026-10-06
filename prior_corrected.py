# -*- coding: utf-8 -*-
"""Step 10 — 先验校正工作点（prior-corrected operating point, m=3, tau=0.3）。
对已发表 checkpoint 在 T1-T4 上施加背景概率 ×3（训练时背景过采样系数的精确
逆）+ 0.3 置信阈，写 results/eval_t*_pc.json 供 Table 2 新行使用。
"""
import importlib
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

ev = importlib.import_module("04_evaluate")
from aqua_common import READS_DIR, RESULTS_DIR, load_json, save_json  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

device = torch.device("cpu")
torch.set_num_threads(8)
model, ckpt = ev.load_model("bacteria", device)
dec_id = load_json(READS_DIR / "label_map.json")["bacteria_decoy_id"]
n_classes = ckpt["n_classes"]
M, TAU = 3.0, 0.3


def read_mean_probs(z):
    flat, off = z["seq"], z["read_offset"].astype(np.int64)
    starts = np.concatenate([[0], off[:-1]])
    R = len(off)
    sums = np.zeros((R, n_classes), dtype=np.float64)
    cnts = np.zeros(R, dtype=np.int64)
    buf, owners = [], []

    def flush():
        nonlocal buf, owners
        if not buf:
            return
        p = ev.predict(model, np.stack(buf), device)
        np.add.at(sums, np.array(owners), p)
        np.add.at(cnts, np.array(owners), 1)
        buf, owners = [], []

    for r, w in ev.iter_read_windows(flat, off):
        if (w == 4).all():
            continue
        buf.append(w)
        owners.append(r)
        if len(buf) >= 4096:
            flush()
    flush()
    return sums / np.maximum(cnts, 1)[:, None]


def apply_op(P):
    L = np.log(np.clip(P, 1e-12, 1))
    L[:, dec_id] += np.log(M)
    pred = L.argmax(1)
    if TAU > 0:
        pred = np.where(np.exp(L.max(1)) >= tau_keep, pred, dec_id)
    return pred


tau_keep = TAU
z1 = np.load(READS_DIR / "t1_seen_bact.npz")
p1 = ev.predict(model, z1["seq"], device)
y1 = z1["label"].astype(int)
z2 = np.load(READS_DIR / "t2_unseen_bact.npz")
p2 = ev.predict(model, z2["seq"], device)
y2 = z2["label"].astype(int)
z3 = np.load(READS_DIR / "t3_reads_bact.npz")
mp3 = read_mean_probs(z3)
y3 = z3["label"].astype(int)
z4 = np.load(READS_DIR / "t4_pond_mix.npz")
mp4 = read_mean_probs(z4)
y4 = z4["label"].astype(int)

for name, y, pred in (
        ("eval_t1_seen_pc", y1, apply_op(p1)),
        ("eval_t2_unseen_pc", y2, apply_op(p2)),
        ("eval_t3_reads_pc", y3, apply_op(mp3)),
        ("eval_t4_pond_pc", y4, apply_op(mp4))):
    save_json({"macro_f1": round(float(f1_score(
                   y, pred, labels=list(range(n_classes)), average="macro",
                   zero_division=0)), 4),
               "acc": round(float(accuracy_score(y, pred)), 4),
               "operating_point": {"m": M, "tau": TAU}},
              RESULTS_DIR / f"{name}.json")
    print(name, "mF1=%.4f acc=%.4f" % (
        f1_score(y, pred, labels=list(range(n_classes)), average="macro",
                 zero_division=0), accuracy_score(y, pred)))
print("STEP10_DONE")
