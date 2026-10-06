"""Step 04 — Evaluate AquaPathNet on all test sets (T1-T6) + runtime benchmark.

Outputs one JSON per test set under results/, plus runtime_benchmark_*.json.
Read-level prediction = mean softmax over the read's 1 kb windows.
Streaming early-warning analysis runs over the shuffled T4 pond mixture.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aqua_common import (CONFIGS_DIR, DATA_ROOT, READS_DIR, RESULTS_DIR,
                         kmer5_features, load_json, load_species_config,
                         save_json, setup_script_logger, append_execution_log)
from aquapathnet import AquaPathNet, AquaPathNetV2, count_params
from importlib import import_module

_seq_to_input = import_module("train").seq_to_input

MODELS_DIR = DATA_ROOT / "models"
SEG_LEN = 1000
MIN_TAIL = 250


def load_model(task: str, device):
    """Load best checkpoint: v2 (dual-branch) if present, else v1."""
    p2 = MODELS_DIR / f"aquapathnet2_{task}.pt"
    path = p2 if p2.exists() else MODELS_DIR / f"aquapathnet_{task}.pt"
    ckpt = torch.load(path, map_location="cpu")
    arch = ckpt.get("arch", "v1")
    if arch == "v1":
        model = AquaPathNet(ckpt["n_classes"], width=ckpt["width"])
    else:
        model = AquaPathNetV2(ckpt["n_classes"], width=ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    model._arch = arch
    model.to(device).eval()
    return model, ckpt


def predict(model, seq: np.ndarray, device, bs: int = 512,
            arch: str | None = None) -> np.ndarray:
    """Window-level softmax probabilities for (N, L) uint8 array."""
    if arch is None:
        arch = getattr(model, "_arch", "v1")
    out = []
    with torch.no_grad():
        for i in range(0, len(seq), bs):
            x = _seq_to_input(seq[i:i + bs]).to(device)
            if arch == "v2":
                xk = torch.from_numpy(kmer5_features(seq[i:i + bs])).to(device)
                out.append(F.softmax(model(x, xk), dim=1).cpu().numpy())
            else:
                out.append(F.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out)


def windows_of_read(flat: np.ndarray, start: int, end: int):
    """Yield 1kb windows of read flat[start:end]; right-pad tail if >= MIN_TAIL."""
    L = end - start
    n_full = L // SEG_LEN
    for w in range(n_full):
        yield flat[start + w * SEG_LEN:start + (w + 1) * SEG_LEN]
    tail = L - n_full * SEG_LEN
    if tail >= MIN_TAIL:
        win = np.full(SEG_LEN, 4, dtype=np.uint8)
        win[:tail] = flat[start + n_full * SEG_LEN:end]
        yield win


def iter_read_windows(flat, off):
    starts = np.concatenate([[0], off[:-1]])
    for r in range(len(off)):
        for w in windows_of_read(flat, int(starts[r]), int(off[r])):
            yield r, w


def read_level_probs(model, z, device, chunk: int = 4096) -> np.ndarray:
    """Mean softmax per read over its windows. Returns (n_reads, n_classes)."""
    flat, off = z["seq"], z["read_offset"]
    n_reads = len(off)
    sums = None
    cnts = np.zeros(n_reads, dtype=np.int64)
    buf, owners = [], []

    def flush():
        nonlocal buf, owners, sums
        if not buf:
            return
        p = predict(model, np.stack(buf), device)
        if sums is None:
            sums = np.zeros((n_reads, p.shape[1]), dtype=np.float64)
        np.add.at(sums, np.array(owners, dtype=np.int64), p)
        np.add.at(cnts, np.array(owners, dtype=np.int64), 1)
        buf, owners = [], []

    for r, w in iter_read_windows(flat, off):
        buf.append(w)
        owners.append(r)
        if len(buf) >= chunk:
            flush()
    flush()
    return sums / np.maximum(cnts, 1)[:, None]


def classify_metrics(y_true, y_pred, slugs) -> dict:
    labels = list(range(len(slugs)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class": {slugs[i]: {"precision": float(prec[i]),
                                 "recall": float(rec[i]), "f1": float(f1[i]),
                                 "support": int(sup[i])} for i in labels},
        "confusion": cm,
    }


def eval_bacteria(log, device) -> None:
    model, ckpt = load_model("bacteria", device)
    inv = {v: k for k, v in ckpt["label_map_subset"].items()}
    slugs = [inv[i] for i in range(ckpt["n_classes"])]
    log.info("bacteria model loaded: %d classes", len(slugs))

    # ---- T1 seen-strain ----
    z = np.load(READS_DIR / "t1_seen_bact.npz")
    p = predict(model, z["seq"], device)
    m = classify_metrics(z["label"].astype(int), p.argmax(1), slugs)
    save_json(m, RESULTS_DIR / "eval_t1_seen.json")
    log.info("T1 seen: acc=%.4f mF1=%.4f", m["acc"], m["macro_f1"])

    # ---- T2 unseen-strain ----
    z = np.load(READS_DIR / "t2_unseen_bact.npz")
    p = predict(model, z["seq"], device)
    m = classify_metrics(z["label"].astype(int), p.argmax(1), slugs)
    save_json(m, RESULTS_DIR / "eval_t2_unseen.json")
    log.info("T2 unseen: acc=%.4f mF1=%.4f", m["acc"], m["macro_f1"])

    # ---- T3 full reads ----
    z = np.load(READS_DIR / "t3_reads_bact.npz")
    pr = read_level_probs(model, z, device)
    m = classify_metrics(z["label"].astype(int), pr.argmax(1), slugs)
    save_json(m, RESULTS_DIR / "eval_t3_reads.json")
    log.info("T3 reads: acc=%.4f mF1=%.4f", m["acc"], m["macro_f1"])

    # ---- T4 pond mixture: read-level + streaming early warning ----
    z = np.load(READS_DIR / "t4_pond_mix.npz")
    pr = read_level_probs(model, z, device)
    y = z["label"].astype(int)
    pred = pr.argmax(1)
    dec_id = load_json(READS_DIR / "label_map.json")["bacteria_decoy_id"]
    binary_t = (y != dec_id).astype(int)
    binary_p = (pred != dec_id).astype(int)
    slugs_t4 = slugs  # 25-class macro over the pond mixture (for tables)
    m_t4 = classify_metrics(y, pred, slugs_t4)
    read_level = {
        "n_reads": int(len(y)),
        "macro_f1_25class": m_t4["macro_f1"],
        "acc_25class": m_t4["acc"],
        "pathogen_recall": float((binary_p[binary_t == 1]).mean()),
        "background_specificity": float((binary_p[binary_t == 0] == 0).mean()),
        "species_correct_of_called": float(
            (pred[binary_t == 1] == y[binary_t == 1]).mean()),
    }
    # streaming: alert when k reads assigned to species s with prob >= theta
    stream = {"rules": []}
    for k in (1, 2, 3, 5):
        for theta in (0.3, 0.5, 0.7):
            n_classes = pr.shape[1]
            cnt = np.zeros(n_classes, dtype=np.int64)
            alerted_at = {}
            for i in range(len(y)):
                s = int(pred[i])
                if s != dec_id and pr[i, s] >= theta:
                    cnt[s] += 1
                    if s not in alerted_at and cnt[s] >= k:
                        alerted_at[s] = i
            truth = set(int(v) for v in np.unique(y) if v != dec_id)
            lat, dets, falses = {}, {}, []
            for s, at in alerted_at.items():
                if s in truth:
                    lat[s] = int(at)
            for s in truth:
                dets[s] = s in lat
            for s in alerted_at:
                if s not in truth:
                    falses.append(s)
            stream["rules"].append({
                "k": k, "theta": theta,
                "detection_rate": float(np.mean([dets[s] for s in truth])),
                "median_latency_reads": (float(np.median([lat[s] for s in lat]))
                                         if lat else None),
                "n_false_alert_species": len(falses),
                "false_alert_species": falses,
            })
    save_json({"read_level": read_level, "streaming": stream},
              RESULTS_DIR / "eval_t4_pond.json")
    log.info("T4 pond: recall=%.4f spec=%.4f", read_level["pathogen_recall"],
             read_level["background_specificity"])

    # ---- T5 robustness grid ----
    z = np.load(READS_DIR / "t5_robustness.npz")
    seq, y, err, L = z["seq"], z["label"].astype(int), z["err"], z["length"]
    pred = predict(model, seq, device).argmax(1)
    grid = {}
    for e in np.unique(err):
        for l in np.unique(L):
            msk = (err == e) & (L == l)
            grid[f"{e:.2f}_{int(l)}"] = {
                "window_macro_f1": float(f1_score(y[msk], pred[msk],
                                                  average="macro",
                                                  zero_division=0)),
                "window_acc": float((pred[msk] == y[msk]).mean()),
            }
    save_json(grid, RESULTS_DIR / "eval_t5_robustness.json")
    log.info("T5 robustness grid: %d cells", len(grid))

    # ---- runtime benchmark ----
    z1 = np.load(READS_DIR / "t1_seen_bact.npz")
    seq = z1["seq"][:4096]
    t0 = time.perf_counter()
    predict(model, seq, device, bs=256)
    dt = time.perf_counter() - t0
    ram = 0.0
    try:
        import psutil
        ram = psutil.Process().memory_info().rss / 1e9
    except Exception:
        pass
    rt = {"params": count_params(model),
          "model_mb_fp32": count_params(model) * 4 / 1e6,
          "segments_per_second_cpu": len(seq) / dt,
          "ms_per_1000bp_segment": 1000 * dt / len(seq),
          "peak_rss_gb": ram, "threads": 8}
    save_json(rt, RESULTS_DIR / "runtime_benchmark_bacteria.json")
    log.info("runtime: %.0f seg/s (%.2f ms/segment)",
             rt["segments_per_second_cpu"], rt["ms_per_1000bp_segment"])


def eval_virus(log, device) -> None:
    model, ckpt = load_model("virus", device)
    inv = {v: k for k, v in ckpt["label_map_subset"].items()}
    slugs = [inv[i] for i in range(ckpt["n_classes"])]
    z = np.load(READS_DIR / "t6_virus_test.npz")
    p = predict(model, z["seq"], device)
    m = classify_metrics(z["label"].astype(int), p.argmax(1), slugs)
    save_json(m, RESULTS_DIR / "eval_t6_virus.json")
    log.info("T6 virus: acc=%.4f mF1=%.4f", m["acc"], m["macro_f1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="bacteria,virus")
    args = ap.parse_args()
    log = setup_script_logger("04_evaluate")
    device = torch.device("cpu")
    torch.set_num_threads(8)
    if "bacteria" in args.tasks:
        eval_bacteria(log, device)
    if "virus" in args.tasks:
        eval_virus(log, device)
    append_execution_log("评估完成 | T1-T6 指标落盘 results/eval_*.json | logs/04_*.log")


if __name__ == "__main__":
    main()
