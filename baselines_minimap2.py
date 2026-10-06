"""Step 05b — minimap2 (map-ont preset) baseline via the official mappy binding.

minimap2 is the aligner used inside ONT's own EPI2ME/WIMP pipelines, so this
is the published-tool alignment baseline. Same protocol as 05_baselines:
best subject by merged query coverage, accept at coverage >= 0.5 and
identity (mlen/blen) >= 0.85, else BACKGROUND.

Outputs: results/eval_t{1,2,3,4}_*_minimap2.json + runtime json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aqua_common import DATA_ROOT, READS_DIR, RESULTS_DIR, load_json, save_json, setup_script_logger, append_execution_log

FASTA_DIR = DATA_ROOT / "fasta"
REF = FASTA_DIR / "ref_bacteria.fna"
COV_MIN = 0.5
IDENT_MIN = 0.85


def classify_reads(aligner, reads: list[tuple[str, str]], classes: dict,
                   dec: int, log):
    """reads: list of (read_id, seq). Returns dict read_id -> class id."""
    per_read = {}
    n_aln = 0
    t0 = time.perf_counter()
    for qid, seq in reads:
        per_sid = {}
        try:
            for hit in aligner.map(seq):
                if hit.is_primary is False and len(per_sid) > 4:
                    continue
                per_sid.setdefault(hit.ctg, []).append(hit)
                n_aln += 1
        except ValueError:
            per_read[qid] = dec
            continue
        best_sid, best_cov, best_ident, best_merged = "", 0.0, 0.0, -1
        qlen = max(1, len(seq))
        for sid, hits in per_sid.items():
            merged, last_end = 0, -1
            ident_sum = len_sum = 0
            for h in sorted(hits, key=lambda x: x.q_st):
                ident_sum += h.mlen
                len_sum += h.blen
                merged += max(0, h.q_en - max(h.q_st, last_end + 1))
                last_end = max(last_end, h.q_en)
            cov = merged / qlen
            ident = ident_sum / max(1, len_sum)
            if merged > best_merged:
                best_sid, best_cov, best_ident, best_merged = (
                    sid, cov, ident, merged)
        slug = best_sid.split("|")[0] if best_sid else ""
        pred = classes.get(slug, dec)
        if not slug or best_cov < COV_MIN or best_ident < IDENT_MIN:
            pred = dec
        per_read[qid] = pred
    dt = time.perf_counter() - t0
    log.info("minimap2: %d reads, %d hits, %.0fs (%.1f reads/s)",
             len(reads), n_aln, dt, len(reads) / dt)
    return per_read, dt


def metrics(y_true, y_pred, n_classes: int) -> dict:
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    return {
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "per_class": {str(i): {"precision": float(prec[i]),
                               "recall": float(rec[i]), "f1": float(f1[i]),
                               "support": int(sup[i])}
                      for i in range(n_classes)},
    }


def main() -> None:
    import mappy
    log = setup_script_logger("05b_minimap2")
    label_map = load_json(READS_DIR / "label_map.json")
    classes = label_map["bacteria"]
    dec = label_map["bacteria_decoy_id"]
    n_classes = dec + 1

    log.info("building minimap2 index from %s ...", REF)
    t0 = time.perf_counter()
    aligner = mappy.Aligner(fn_idx_in=str(REF), preset="map-ont")
    if not aligner:
        raise RuntimeError("minimap2 index failed")
    log.info("index ready in %.0fs", time.perf_counter() - t0)

    rt = {"index_seconds": round(time.perf_counter() - t0, 1)}
    try:
        import psutil
        rt["index_rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        pass

    for set_name, read_mode in (("t1_seen_bact", False),
                                ("t2_unseen_bact", False),
                                ("t3_reads_bact", True),
                                ("t4_pond_mix", True)):
        raw = np.load(READS_DIR / f"{set_name}.npz")
        z = {k: raw[k] for k in raw.files}
        reads = []
        if read_mode:
            flat, off = z["seq"], z["read_offset"]
            starts = np.concatenate([[0], off[:-1]])
            for r in range(len(off)):
                seq = flat[starts[r]:off[r]]
                if len(seq) >= 5:
                    reads.append((f"read{r}",
                                  "".join("ACGTN"[int(b)] for b in seq)))
                else:
                    reads.append((f"read{r}", ""))
        else:
            for i in range(len(z["label"])):
                reads.append((f"seg{i}",
                              "".join("ACGTN"[int(b)] for b in z["seq"][i])))
        per_read, dt = classify_reads(aligner, reads, classes, dec, log)
        y_pred = [per_read[f"seg{i}" if not read_mode else f"read{i}"]
                  for i in range(len(z["label"]))]
        m = metrics(z["label"].astype(int), np.array(y_pred), n_classes)
        m["runtime_seconds"] = round(dt, 1)
        m["reads_per_second"] = round(len(reads) / dt, 1)
        save_json(m, RESULTS_DIR / f"eval_{set_name}_minimap2.json")
        rt[set_name] = {"seconds": round(dt, 1),
                        "reads_per_second": round(len(reads) / dt, 1)}
        log.info("minimap2 %s: acc=%.4f mF1=%.4f", set_name, m["acc"],
                 m["macro_f1"])
    save_json(rt, RESULTS_DIR / "runtime_benchmark_minimap2.json")
    append_execution_log("minimap2 基线完成 | results/eval_*_minimap2.json")


if __name__ == "__main__":
    main()
