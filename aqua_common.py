"""AquaPathID common utilities: logging, paths, FASTA IO, reproducibility.

Project layout (see README.md):
  PROJECT_ROOT = repository root (directory containing this file)
  DATA_ROOT    = os.environ.get("AQUAPATHID_DATA", "./AquaPathID_data")
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("AQUAPATHID_DATA", "./AquaPathID_data"))
REFS_DIR = DATA_ROOT / "refs"
READS_DIR = DATA_ROOT / "reads"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
FIGURES_DIR = PROJECT_ROOT / "figures"

SEED = 42

# Standard base encoding. N -> 4.
BASES = "ACGTN"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
NUM_BASES = 5  # A,C,G,T,N


def utc_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def setup_script_logger(script_name: str) -> logging.Logger:
    """Log to logs/<script>_<timestamp>.log (UTF-8) and stdout (ASCII-safe)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{script_name}_{ts}.log"
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logger.info("log file: %s", log_file)
    return logger


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_species_config() -> dict:
    with open(CONFIGS_DIR / "species.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_fasta_records(path: Path):
    """Yield (header, seq_str) from plain or gzipped FASTA. Streams, low memory."""
    op = gzip.open if str(path).endswith(".gz") else open
    header, chunks = None, []
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line.upper())
        if header is not None:
            yield header, "".join(chunks)


def seq_to_array(seq: str) -> np.ndarray:
    """Convert DNA string to uint8 base-index array (A=0,C=1,G=2,T=3,N=4)."""
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    lut = np.full(256, 4, dtype=np.uint8)
    for i, b in enumerate(BASES):
        lut[ord(b)] = i
    lut[ord("U")] = 3
    return lut[arr]


def append_execution_log(line: str) -> None:
    """Append one row to EXECUTION_LOG.md (全程留痕)."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = PROJECT_ROOT / "EXECUTION_LOG.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"| {stamp} | {line} |\n")


def reverse_complement(seq: np.ndarray) -> np.ndarray:
    m = np.array([3, 2, 1, 0, 4], dtype=np.uint8)  # A<->T, C<->G, N->N
    return m[seq[::-1]]


def read_length(rng: np.random.Generator, rl_params: dict) -> int:
    n = int(round(np.exp(rng.normal(np.log(rl_params["median_bp"]),
                                    rl_params["sigma"]))))
    return int(np.clip(n, rl_params["min_bp"], rl_params["max_bp"]))


POW5 = (5 ** np.arange(4, -1, -1)).astype(np.int64)
KMER5_DIM = 5 ** 5  # 3125


def kmer5_features(seqs: np.ndarray) -> np.ndarray:
    """(B, L) uint8 base-code arrays -> (B, 3125) normalised 5-mer counts."""
    win = np.lib.stride_tricks.sliding_window_view(
        seqs.astype(np.int64), 5, axis=-1)
    codes = win @ POW5
    out = np.zeros((len(seqs), KMER5_DIM), dtype=np.float32)
    for i in range(len(seqs)):
        c = np.bincount(codes[i], minlength=KMER5_DIM)
        out[i] = c / max(1, c.sum())
    return out


def pad_or_trim(seg: np.ndarray, seg_len: int) -> np.ndarray:
    """Force a segment to exactly seg_len by right-padding with N (4) or trimming
    (error injection changes length: deletions shrink, insertions grow)."""
    if len(seg) == seg_len:
        return seg
    if len(seg) > seg_len:
        return seg[:seg_len]
    out = np.full(seg_len, 4, dtype=np.uint8)
    out[:len(seg)] = seg
    return out


def inject_errors(seq: np.ndarray, total_err: float, p: dict,
                  rng: np.random.Generator) -> np.ndarray:
    """Apply ONT-like errors to a uint8 base-code array; returns new array.

    seq uses codes 0..4 (A,C,G,T,N); N positions are never mutated (unknown).
    Error-type ratio and homopolymer behaviour from configs/sim_params.json
    (cf. published ONT R9.4-Guppy / R10.4-Dorado characteristics).
    """
    ratio = p["error_type_ratio"]
    p_sub = total_err * ratio["mismatch"]
    p_ins = total_err * ratio["insert"]
    p_del = total_err * ratio["delete"]
    n = len(seq)

    # --- deletions (homopolymer-aware) ---
    del_p = np.full(n, p_del, dtype=np.float32)
    d = np.diff(seq)
    run = np.ones(n, dtype=np.int32)
    if n > 1:
        idx = np.arange(1, n)
        run[1:] = idx - np.maximum.accumulate(np.where(d != 0, idx, 0)) + 1
    hp = (run >= p["homopolymer_run_min"]) & (seq != 4)
    del_p[hp] = np.minimum(p_del * p["homopolymer_del_multiplier"], 0.9)
    keep = rng.random(n) >= del_p
    if not keep.all():
        seq = seq[keep]

    # --- insertions ---
    n2 = len(seq)
    ins_mask = rng.random(n2) < p_ins
    if ins_mask.any():
        extra = rng.integers(0, 4, size=int(ins_mask.sum()), dtype=np.uint8)
        seq = np.insert(seq.astype(np.uint8), np.nonzero(ins_mask)[0], extra)

    # --- substitutions (never into N) ---
    sub_mask = (rng.random(len(seq)) < p_sub) & (seq != 4)
    if sub_mask.any():
        r = rng.integers(0, 4, size=len(seq), dtype=np.uint8)
        seq[sub_mask] = r[sub_mask]
    return seq


class Timer:
    def __init__(self, label: str, logger: logging.Logger | None = None):
        self.label = label
        self.logger = logger
        self.t0 = time.perf_counter()

    def done(self) -> float:
        dt = time.perf_counter() - self.t0
        msg = f"[timer] {self.label}: {dt:.1f}s"
        if self.logger:
            self.logger.info(msg)
        return dt
