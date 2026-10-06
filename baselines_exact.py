"""Step 05 鈥?Baseline methods: (A) Kraken2-style exact minimizer-LCA k-mer
classifier (in-house, NumPy) and (B) BLASTN megablast alignment (NCBI BLAST+).

Both baselines are evaluated on the same test sets as AquaPathNet:
  T1/T2 (1 kb segments) and T3/T4 (variable-length reads).

(A) Index: k=31 exact 2-bit codes, minimizer sampling (w=11 window), built ONLY
    from training genomes. k-mers shared by >1 class collapse to "ambiguous"
    (virtual root, dropped from votes) 鈥?i.e., species-level LCA collapse.
    A read is assigned to the class with most distinct minimizer votes if
    votes >= min_votes, else BACKGROUND.

(B) BLASTN: megablast against the concatenated training-genome reference.
    A read is assigned to the species of its best HSP if coverage >= 0.5 and
    identity >= 0.85, else BACKGROUND.

Outputs: results/eval_<set>_<baseline>.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import f1_score, precision_recall_fscore_support
from scipy.ndimage import minimum_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aqua_common import (DATA_ROOT, READS_DIR, RESULTS_DIR, iter_fasta_records,
                         load_json, load_species_config, save_json,
                         setup_script_logger, append_execution_log)

K = 31
W = 11  # minimizer window
MIN_VOTES = 3
BLAST_ROOT = (Path(os.environ.get("AQUAPATHID_TOOLS", "./tools"))
              / "ncbi-blast-2.17.0+" / "bin")
BLAST_EXE = "blastn.exe" if os.name == "nt" else "blastn"
FASTA_DIR = DATA_ROOT / "fasta"
GOLD = np.uint64(0x9E3779B97F4A7C15)
POW4 = (np.uint64(4) ** np.arange(K - 1, -1, -1, dtype=np.uint64))


def kmer_codes(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """31-mer rolling codes (uint64, wrapped) + validity mask (no N inside)."""
    n = len(arr)
    if n < K:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=bool)
    codes = np.zeros(n - K + 1, dtype=np.uint64)
    bad = np.zeros(n - K + 1, dtype=bool)
    win = sliding_window_view(arr, K)
    for j in range(K):
        codes *= np.uint64(4)
        codes += win[:, j].astype(np.uint64)
        bad |= (win[:, j] == 4)
    return codes, ~bad


def minimizers(codes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Local-minimum minimizer selection over a W-window of k-mer codes.

    scipy's minimum_filter1d casts to float64, which destroys uint64 ordering,
    so hash values are masked to 63 bits and compared as int64.
    """
    if len(codes) < W:
        return codes[valid]
    h = ((codes * GOLD) & np.uint64(0x7FFFFFFFFFFFFFFF)).astype(np.int64)
    mins = minimum_filter1d(h, size=W, mode="nearest")
    sel = (h == mins) & valid
    return codes[sel]


def load_class_map(task: str) -> tuple[dict, int]:
    label_map = load_json(READS_DIR / "label_map.json")
    classes = label_map["bacteria" if task == "bacteria" else "virus"]
    dec = label_map["bacteria_decoy_id" if task == "bacteria" else "virus_decoy_id"]
    return classes, dec


def build_kmer_index(task: str, log) -> tuple[np.ndarray, np.ndarray]:
    """Build sorted (codes, labels) minimizer index from TRAIN genomes only."""
    idx_file = READS_DIR / f"kmer_index_{task}.npz"
    if idx_file.exists():
        z = np.load(idx_file)
        log.info("k-mer index loaded: %d minimizers", len(z["codes"]))
        return z["codes"], z["labels"]
    split = load_json(READS_DIR / "split_map.json")
    genome_meta = load_json(READS_DIR / "genome_meta.json")
    classes, dec = load_class_map(task)
    dec_slugs = set(d["slug"] for d in load_species_config()["decoys"])
    codes_all, labels_all = [], []
    t0 = time.perf_counter()
    for m in genome_meta:
        info = split[m["key"]]
        slug = info["slug"]
        if task == "bacteria":
            label = classes.get(slug, dec)
        else:
            label = classes.get(slug, dec)
        if slug in dec_slugs and task == "virus":
            label = dec
        if not info["role"].startswith("train"):
            continue
        arr = np.load(READS_DIR / "genome_cache.npz")[m["key"]]
        codes, valid = kmer_codes(arr)
        mins = minimizers(codes, valid)
        codes_all.append(mins)
        labels_all.append(np.full(len(mins), label, dtype=np.int16))
        log.info("indexed %-30s %d minimizers", slug, len(mins))
    codes = np.concatenate(codes_all)
    labels = np.concatenate(labels_all)
    order = np.argsort(codes, kind="stable")
    codes, labels = codes[order], labels[order]
    # LCA collapse: k-mer shared by >=2 distinct classes -> ambiguous (-1)
    uniq, starts, counts = np.unique(codes, return_index=True, return_counts=True)
    lab = labels[starts]
    multi = counts > 1
    # check whether any multi-mapping k-mer has differing labels
    amb = np.zeros(len(uniq), dtype=bool)
    if multi.any():
        # compare first and last label within each run
        ends = starts + counts - 1
        amb[multi] = labels[ends[multi]] != lab[multi]
    lab[amb] = -1
    keep = ~amb
    np.savez_compressed(idx_file, codes=uniq[keep], labels=lab[keep])
    log.info("k-mer index built: %d unique minimizers (%.0fs), %.1f%% ambiguous",
             len(uniq), time.perf_counter() - t0, 100 * amb.mean())
    return uniq[keep], lab[keep]


def classify_segments_lca(idx_codes, idx_labels, X: np.ndarray) -> np.ndarray:
    """Classify (N, 1000) uint8 segments. Returns class ids (BACKGROUND=dec)."""
    n_classes = int(idx_labels.max()) + 2
    out = np.empty(len(X), dtype=np.int64)
    for i in range(len(X)):
        codes, valid = kmer_codes(X[i])
        mins = minimizers(codes, valid)
        out[i] = _vote(idx_codes, idx_labels, mins, n_classes)
    return out


def classify_reads_lca(idx_codes, idx_labels, flat, off) -> np.ndarray:
    n_classes = int(idx_labels.max()) + 2
    starts = np.concatenate([[0], off[:-1]])
    out = np.empty(len(off), dtype=np.int64)
    for r in range(len(off)):
        codes, valid = kmer_codes(flat[starts[r]:off[r]])
        mins = minimizers(codes, valid)
        out[r] = _vote(idx_codes, idx_labels, mins, n_classes)
    return out


def _vote(idx_codes, idx_labels, mins, n_classes) -> int:
    if len(mins) == 0:
        return n_classes - 1  # BACKGROUND
    pos = np.searchsorted(idx_codes, mins)
    pos = np.clip(pos, 0, len(idx_codes) - 1)
    hit = idx_codes[pos] == mins
    labels = idx_labels[pos[hit]]
    labels = labels[labels >= 0]
    if len(labels) == 0:
        return n_classes - 1
    votes = np.bincount(labels, minlength=n_classes - 1)
    top = int(votes.argmax())
    return top if votes[top] >= MIN_VOTES else n_classes - 1


# ------------------------------------------------------------- FASTA export
def export_segments_fasta(z, path: Path, inv_slug: dict):
    seq = z["seq"]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i in range(len(seq)):
            s = "".join("ACGTN"[b] for b in seq[i])
            f.write(f">seg{i}|label={z['label'][i]}\n{s}\n")


def export_reads_fasta(z, path: Path):
    flat, off = z["seq"], z["read_offset"]
    starts = np.concatenate([[0], off[:-1]])
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in range(len(off)):
            s = "".join("ACGTN"[int(b)] for b in flat[starts[r]:off[r]])
            f.write(f">read{r}|label={z['label'][r]}\n{s}\n")


def build_reference_fasta(path: Path, log):
    if path.exists():
        log.info("reference fasta exists: %s", path)
        return
    split = load_json(READS_DIR / "split_map.json")
    genome_meta = load_json(READS_DIR / "genome_meta.json")
    cache = np.load(READS_DIR / "genome_cache.npz")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for m in genome_meta:
            if not split[m["key"]]["role"].startswith("train"):
                continue
            arr = cache[m["key"]]
            s = "".join("ACGTN"[int(b)] for b in arr)
            f.write(f">{m['slug']}|{m['assembly']}\n")
            for i in range(0, len(s), 70):
                f.write(s[i:i + 70] + "\n")
    log.info("reference fasta written: %s (%.0f MB)", path,
             path.stat().st_size / 1e6)


def metrics(y_true, y_pred, n_classes: int) -> dict:
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


def subset_z(z: dict, pick: np.ndarray) -> dict:
    """Subset an npz dict, handling variable-length read layout correctly."""
    z = {k: z[k] for k in z.files} if hasattr(z, "files") else dict(z)
    if "read_offset" in z:
        off = z["read_offset"].astype(np.int64)
        starts = np.concatenate([[0], off[:-1]])
        lens = off - starts
        flat_idx = np.concatenate(
            [np.arange(starts[i], off[i]) for i in pick])
        return {"seq": z["seq"][flat_idx],
                "read_offset": np.cumsum(lens[pick]),
                "label": z["label"][pick]}
    return {k: v[pick] for k, v in z.items()}


def run_blast(log, task: str, subset: int | None = 4000):
    """BLASTN baseline on T1/T3 (subsampled for tractability)."""
    ref = FASTA_DIR / f"ref_{task}.fna"
    build_reference_fasta(ref, log)
    blastn = BLAST_ROOT / BLAST_EXE
    if not blastn.exists():
        log.warning("BLAST+ not found at %s -> skipping BLAST baseline", blastn)
        return
    classes, dec = load_class_map(task)
    n_classes = dec + 1
    for set_name, read_mode in (("t1_seen_bact", False),
                                ("t2_unseen_bact", False),
                                ("t3_reads_bact", True)):
        z_raw = np.load(READS_DIR / f"{set_name}.npz")
        z = {k: z_raw[k] for k in z_raw.files}
        if subset and len(z["label"]) > subset:
            rng = np.random.default_rng(7)
            pick = np.sort(rng.choice(len(z["label"]), subset, replace=False))
            z = subset_z(z, pick)
        q = FASTA_DIR / f"{set_name}_blast_query.fna"
        if read_mode:
            export_reads_fasta(z, q)
        else:
            export_segments_fasta(z, q, None)
        out = RESULTS_DIR / f"blast_{set_name}.tsv"
        if not out.exists():
            t0 = time.perf_counter()
            with open(out, "w") as fo:
                subprocess.run(
                    [str(blastn), "-task", "blastn", "-word_size", "11",
                     "-query", str(q),
                     "-subject", str(ref), "-outfmt",
                     "6 qseqid sseqid qlen qstart qend nident length",
                     "-num_threads", "8", "-evalue", "1e-10",
                     "-dust", "no"],
                    stdout=fo, check=True)
            log.info("BLAST %s done in %.0fs", set_name,
                     time.perf_counter() - t0)
        # aggregate HSPs per (query, subject): ONT-style error-containing reads
        # align as several HSPs; single-HSP coverage would reject most true hits
        hsp, qlen_of = {}, {}
        with open(out, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 7:
                    continue
                # query ids carry a "|label=N" suffix from the FASTA export;
                # strip it so lookups by "seg{i}"/"read{i}" match
                qid = parts[0].split("|")[0]
                sid, qlen, qs, qe, nident, length = parts[1:]
                qlen_of[qid] = int(qlen)
                hsp.setdefault(qid, {}).setdefault(sid, []).append(
                    (int(qs), int(qe), int(nident), int(length)))
        best = {}
        for qid, per_sid in hsp.items():
            best_sid, best_cov, best_ident, best_score = "", 0.0, 0.0, -1
            for sid, hsps in per_sid.items():
                merged, last_end = 0, -1
                ident_sum = len_sum = 0
                for qs, qe, nident, length in sorted(hsps):
                    ident_sum += nident
                    len_sum += length
                    merged += max(0, qe - max(qs, last_end + 1))
                    last_end = max(last_end, qe)
                cov = merged / max(1, qlen_of[qid])
                ident = ident_sum / max(1, len_sum)
                if merged > best_score:
                    best_sid, best_cov, best_ident, best_score = (
                        sid, cov, ident, merged)
            if best_sid:
                best[qid] = (best_sid, best_cov, best_ident)
        y_true, y_pred = [], []
        n_read = len(z["label"])
        for i in range(n_read):
            sid, cov, ident = best.get(f"seg{i}" if not read_mode else f"read{i}", ("", 0.0, 0.0))
            slug = sid.split("|")[0] if sid else ""
            classes, dec_ = load_class_map(task)
            pred = classes.get(slug, dec_)
            if not slug or cov < 0.5 or ident < 0.85:
                pred = dec_
            y_true.append(int(z["label"][i]))
            y_pred.append(pred)
        m = metrics(np.array(y_true), np.array(y_pred), n_classes)
        save_json(m, RESULTS_DIR / f"eval_{set_name}_blast.json")
        log.info("BLAST %s: acc=%.4f mF1=%.4f", set_name, m["acc"], m["macro_f1"])


def run_kmerlog(log, task: str, n_train: int = 40000):
    """(C) KmerLog: 5-mer composition + multinomial logistic regression.

    A strong, fast, interpretable baseline: species are largely separable from
    k-mer composition, so this shows how much the CNN adds beyond composition.
    Training segments come from the same on-the-fly sampler as the model.
    """
    from sklearn.linear_model import LogisticRegression
    from importlib import import_module
    from aqua_common import CONFIGS_DIR, kmer5_features, load_species_config

    train_mod = import_module("train")

    def feats(seqs: np.ndarray) -> np.ndarray:
        return kmer5_features(seqs)

    log.info("KmerLog: sampling %d training segments ...", n_train)
    genomes = {k: v for k, v in np.load(READS_DIR / "genome_cache.npz").items()}
    split = load_json(READS_DIR / "split_map.json")
    meta = load_json(READS_DIR / "genome_meta.json")
    sampler = train_mod.GenBankSampler(
        genomes, split, meta, load_species_config(), task,
        load_json(CONFIGS_DIR / "sim_params.json"), {})
    Xs, ys = [], []
    got = 0
    while got < n_train:
        X, y = sampler.batch(256)
        Xs.append(feats(X))
        ys.append(y)
        got += len(y)
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
    clf = LogisticRegression(max_iter=500, C=10.0)
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    log.info("KmerLog fitted in %.0fs", time.perf_counter() - t0)

    _, dec = load_class_map(task)
    n_classes = dec + 1
    for set_name in ("t1_seen_bact", "t2_unseen_bact"):
        z = np.load(READS_DIR / f"{set_name}.npz")
        pred = clf.predict(feats(z["seq"]))
        m = metrics(z["label"].astype(int), pred, n_classes)
        save_json(m, RESULTS_DIR / f"eval_{set_name}_kmerlog.json")
        log.info("KmerLog %s: acc=%.4f mF1=%.4f", set_name, m["acc"],
                 m["macro_f1"])
    for set_name in ("t3_reads_bact", "t4_pond_mix"):
        z = np.load(READS_DIR / f"{set_name}.npz")
        flat, off = z["seq"], z["read_offset"]
        starts = np.concatenate([[0], off[:-1]])
        X = np.zeros((len(off), 5 ** 5))
        _, dec_ = load_class_map(task)
        short = []
        for r in range(len(off)):
            seg = flat[starts[r]:off[r]]
            if len(seg) < 5:
                short.append(r)
                continue
            c = np.bincount((np.lib.stride_tricks.sliding_window_view(
                seg, 5).astype(np.int64) @ __import__("aqua_common").POW5),
                minlength=5 ** 5)
            X[r] = c / max(1, c.sum())
        pred = clf.predict(X)
        pred[short] = dec_
        m = metrics(z["label"].astype(int), pred, n_classes)
        save_json(m, RESULTS_DIR / f"eval_{set_name}_kmerlog.json")
        log.info("KmerLog %s: acc=%.4f mF1=%.4f", set_name, m["acc"],
                 m["macro_f1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="bacteria", choices=["bacteria", "virus"])
    ap.add_argument("--baselines", default="lca,blast,kmerlog")
    args = ap.parse_args()
    log = setup_script_logger(f"05_baselines_{args.task}")
    classes, dec = load_class_map(args.task)
    n_classes = dec + 1
    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    baselines = args.baselines.split(",")

    if "lca" in baselines:
        idx_codes, idx_labels = build_kmer_index(args.task, log)
        t0 = time.perf_counter()
        z = np.load(READS_DIR / "t1_seen_bact.npz")
        pred = classify_segments_lca(idx_codes, idx_labels, z["seq"])
        m = metrics(z["label"].astype(int), pred, n_classes)
        m["runtime"] = {"segments": len(pred),
                        "seconds": round(time.perf_counter() - t0, 1)}
        save_json(m, RESULTS_DIR / f"eval_t1_seen_lca.json")
        log.info("LCA T1: acc=%.4f mF1=%.4f", m["acc"], m["macro_f1"])

        t0 = time.perf_counter()
        z = np.load(READS_DIR / "t2_unseen_bact.npz")
        pred = classify_segments_lca(idx_codes, idx_labels, z["seq"])
        m = metrics(z["label"].astype(int), pred, n_classes)
        save_json(m, RESULTS_DIR / "eval_t2_unseen_lca.json")
        log.info("LCA T2: acc=%.4f mF1=%.4f", m["acc"], m["macro_f1"])

        t0 = time.perf_counter()
        z = np.load(READS_DIR / "t3_reads_bact.npz")
        pred = classify_reads_lca(idx_codes, idx_labels, z["seq"],
                                  z["read_offset"])
        m = metrics(z["label"].astype(int), pred, n_classes)
        save_json(m, RESULTS_DIR / "eval_t3_reads_lca.json")
        log.info("LCA T3: acc=%.4f mF1=%.4f", m["acc"], m["macro_f1"])

        t0 = time.perf_counter()
        z = np.load(READS_DIR / "t4_pond_mix.npz")
        pred = classify_reads_lca(idx_codes, idx_labels, z["seq"],
                                  z["read_offset"])
        m = metrics(z["label"].astype(int), pred, n_classes)
        save_json(m, RESULTS_DIR / "eval_t4_pond_lca.json")
        log.info("LCA T4: acc=%.4f mF1=%.4f", m["acc"], m["macro_f1"])

    if "blast" in baselines:
        run_blast(log, args.task)

    if "kmerlog" in baselines:
        run_kmerlog(log, args.task)

    append_execution_log(
        f"鍩虹嚎 {args.task} ({','.join(baselines)}) 瀹屾垚 | results/eval_*_lca.json, eval_*_blast.json")


if __name__ == "__main__":
    main()


