"""Step 01/02 — ONT read simulator + dataset builder for AquaPathID.

Simulator design (parameterised in configs/sim_params.json):
  * Error types: substitutions / insertions / deletions in a fixed ratio
    (default 55/22/23, cf. published ONT R9.4-Guppy and R10.4-Dorado profiles).
  * Homopolymer runs (>= 4 identical bases) have `homopolymer_del_multiplier`
    higher deletion probability (known ONT artefact).
  * Read lengths ~ lognormal (median 8 kb, sigma 1.0) as typical for ONT.
  * A "segment" (default 1000 bp) is the classification unit: training samples
    windows from genomes and injects an error profile per segment; test sets
    store fixed reproducible arrays (numpy uint8 codes: A0 C1 G2 T3 N4).

Outputs ($AQUAPATHID_DATA/reads/):
  genome_cache.npz    -- all genomes as packed uint8 arrays (RAM-friendly reload)
  split_map.json      -- train/val/unseen-strain assignment per genome
  label_map.json      -- class definitions for the two tasks (bacteria / virus)
  t0_val_bact.npz, t0_val_virus.npz
  t1_seen_bact.npz    -- seen-strain test (unseen positions)
  t2_unseen_bact.npz  -- held-out strain test
  t3_reads_bact.npz   -- variable-length full reads
  t4_pond_mix.npz     -- pond-realistic mixture (95% background / 5% pathogen reads)
  t5_robustness.npz   -- error-rate x read-length grid
  t6_virus_test.npz   -- virus task test
Each npz: seq (uint8 packed codes), label (int16), err (float32), src (int32 index
into meta['source']), plus meta.json arrays; reproducible with SEED.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aqua_common import (READS_DIR, REFS_DIR, RESULTS_DIR, Timer, inject_errors,
                         load_json, load_species_config, pad_or_trim,
                         read_length, reverse_complement, save_json,
                         seq_to_array, set_all_seeds, setup_script_logger,
                         append_execution_log)

READS_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- simulator
def sample_segments(genomes: dict, keys: list[str], weights: np.ndarray,
                    n: int, seg_len: int, err_sampler, p: dict,
                    rng: np.random.Generator, rc_aug: bool) -> tuple:
    """Sample n segments from genomes with error injection. Returns (X, meta)."""
    out = np.empty((n, seg_len), dtype=np.uint8)
    meta = {"src": np.empty(n, dtype=np.int32),
            "err": np.empty(n, dtype=np.float32),
            "start": np.empty(n, dtype=np.int64)}
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    chosen = rng.choice(len(keys), size=n, p=weights)
    for i in range(n):
        key = keys[chosen[i]]
        arr = genomes[key]
        if len(arr) <= seg_len + 10:
            start = 0
            seg = arr[:seg_len].copy()
            seg = np.pad(seg, (0, seg_len - len(seg)), constant_values=4)
        else:
            start = int(rng.integers(0, len(arr) - seg_len))
            seg = arr[start:start + seg_len].copy()
        err = err_sampler(rng)
        seg = inject_errors(seg, err, p, rng)
        seg = pad_or_trim(seg, seg_len)
        if rc_aug and rng.random() < 0.5:
            seg = reverse_complement(seg)
        out[i] = seg
        meta["src"][i] = chosen[i]
        meta["err"][i] = err
        meta["start"][i] = start
    return out, meta


def sample_reads(genomes: dict, keys: list[str], n_reads: int, p: dict,
                 err_total: float, rng: np.random.Generator) -> tuple:
    """Sample n full-length reads (variable length) from uniformly chosen keys."""
    seqs, srcs = [], []
    chosen = rng.integers(0, len(keys), size=n_reads)
    for i in range(n_reads):
        arr = genomes[keys[chosen[i]]]
        L = min(read_length(rng, p["read_length_lognormal"]), len(arr))
        if L >= len(arr):
            start = 0
        else:
            start = int(rng.integers(0, len(arr) - L))
        seg = inject_errors(arr[start:start + L].copy(), err_total, p, rng)
        if p["reverse_comp_augmentation"] and rng.random() < 0.5:
            seg = reverse_complement(seg)
        seqs.append(seg)
        srcs.append(chosen[i])
    flat = np.concatenate(seqs)
    offsets = np.cumsum([0] + [len(s) for s in seqs], dtype=np.int64)
    return flat, offsets, np.array(srcs, dtype=np.int32)


# ------------------------------------------------------------ genome cache
def load_or_build_cache(manifest: dict, log) -> dict:
    cache_file = READS_DIR / "genome_cache.npz"
    if cache_file.exists():
        log.info("loading existing genome cache ...")
        with np.load(cache_file) as z:
            return {k: z[k] for k in z.files}
    genomes, meta = {}, []
    for g in manifest["genomes"]:
        slug, f = g["slug"], Path(g["file"])
        if not f.exists():
            log.warning("missing file for %s: %s", slug, f)
            continue
        from aqua_common import iter_fasta_records
        seqs = [seq_to_array(s) for _, s in iter_fasta_records(f)]
        arr = np.concatenate(seqs) if seqs else np.empty(0, dtype=np.uint8)
        if len(arr) < 5000 and g["kind"] == "bacterium":
            log.warning("tiny genome skipped %s %s (%d bp)", slug, g["assembly"], len(arr))
            continue
        key = f"{slug}#{g['assembly']}"
        genomes[key] = arr
        meta.append({**g, "key": key, "len_bp": int(len(arr))})
    np.savez_compressed(cache_file, **genomes)
    save_json(meta, READS_DIR / "genome_meta.json")
    log.info("genome cache built: %d genomes", len(genomes))
    return genomes


def build_split_and_labels(genome_meta: list[dict], cfg_species: dict) -> tuple:
    """Assign train/val/unseen per genome and build label maps for both tasks."""
    by_species: dict[str, list[dict]] = {}
    for m in genome_meta:
        by_species.setdefault(m["slug"], []).append(m)
    split = {}
    for slug, gms in by_species.items():
        gms.sort(key=lambda x: x["assembly"])
        n = len(gms)
        if slug in [d["slug"] for d in cfg_species["decoys"]]:
            role = ["train"] + ["test_unseen"] * (n - 1) if n > 1 else ["train"]
            if n >= 2:
                role = ["train", "test_unseen"] + ["test_unseen"] * (n - 2)
        else:
            if n >= 4:
                role = ["train", "train", "val", "test_unseen"] + ["train"] * (n - 4)
            elif n == 3:
                role = ["train", "train", "val_unseen"]
            elif n == 2:
                role = ["train", "val_unseen"]
            else:
                role = ["train"]
        for gm, r in zip(gms, role):
            split[gm["key"]] = {"role": r, "slug": slug,
                                "true_unseen": r == "test_unseen"}
    bact_classes = [p["slug"] for p in cfg_species["pathogens"]
                    if p["group"] == "bacterium"]
    virus_classes = [p["slug"] for p in cfg_species["pathogens"]
                     if p["group"] == "virus"]
    label_map = {
        "bacteria": {s: i for i, s in enumerate(bact_classes)},
        "bacteria_decoy_id": len(bact_classes),
        "virus": {s: i for i, s in enumerate(virus_classes)},
        "virus_decoy_id": len(virus_classes),
    }
    return split, label_map


def key_by_role(genome_meta, split, slugs, roles):
    keys, w = [], []
    for m in genome_meta:
        s = split[m["key"]]
        if s["slug"] in slugs and s["role"] in roles:
            keys.append(m["key"])
    return keys


def main() -> None:
    log = setup_script_logger("01_simulate_reads")
    set_all_seeds(42)
    p = load_json(__import__("aqua_common").CONFIGS_DIR / "sim_params.json")
    manifest = load_json(RESULTS_DIR / "genome_manifest.json")
    cfg_species = load_species_config()

    t = Timer("build genome cache", log)
    genomes = load_or_build_cache(manifest, log)
    genome_meta = load_json(READS_DIR / "genome_meta.json")
    t.done()

    split, label_map = build_split_and_labels(genome_meta, cfg_species)
    save_json(split, READS_DIR / "split_map.json")
    save_json(label_map, READS_DIR / "label_map.json")

    bact = label_map["bacteria"]
    dec_id = label_map["bacteria_decoy_id"]
    dec_slugs = [d["slug"] for d in cfg_species["decoys"]]
    virus = label_map["virus"]
    vdec_id = label_map["virus_decoy_id"]
    seg_len = p["segment_len"]
    ts = p["test_sets"]
    lo, hi = p["train_total_error_range"]

    def err_uniform(rng):
        return float(rng.uniform(lo, hi))

    def err_fixed(e):
        return (lambda rng: e)

    def save_npz(name, X, y, meta):
        np.savez_compressed(READS_DIR / name, seq=X, label=y, **meta)
        log.info("saved %-22s X=%s labels=%d uniq=%d", name, X.shape, len(y),
                 len(np.unique(y)))

    # ---- T0 validation ----
    for task, classes, dec, task_slugs in [
        ("bact", bact, dec_id, None), ("virus", virus, vdec_id, None)]:
        cls_keys = {}
        n_per = ts["t0_val_segments_per_pathogen"]
        for slug in classes:
            keys = key_by_role(genome_meta, split, [slug], ["val", "val_unseen"])
            if not keys:
                keys = key_by_role(genome_meta, split, [slug], ["train"])
            cls_keys[slug] = keys
        Xl, yl, ml = [], [], []
        rng = np.random.default_rng(420)
        for slug, keys in cls_keys.items():
            arr_keys = list(keys)
            w = np.array([1.0] * len(arr_keys))
            X, meta = sample_segments(genomes, arr_keys, w, n_per, seg_len,
                                      err_uniform, p, rng, True)
            Xl.append(X)
            yl.append(np.full(n_per, classes[slug], dtype=np.int16))
            ml.append({k: v for k, v in meta.items()})
        dkeys = key_by_role(genome_meta, split, dec_slugs, ["train", "val"])
        n_dec = n_per
        X, meta = sample_segments(genomes, dkeys, np.ones(len(dkeys)), n_dec,
                                  seg_len, err_uniform, p, rng, True)
        Xl.append(X)
        yl.append(np.full(n_dec, dec, dtype=np.int16))
        ml.append(meta)
        save_npz(f"t0_val_{task}.npz", np.concatenate(Xl),
                 np.concatenate(yl), {k: np.concatenate([m[k] for m in ml])
                                      for k in ml[0]})

    # ---- T1 seen-strain test ----
    rng = np.random.default_rng(421)
    Xl, yl, ml = [], [], []
    for slug in bact:
        keys = key_by_role(genome_meta, split, [slug], ["train"])
        X, meta = sample_segments(genomes, keys, np.ones(len(keys)),
                                  ts["t1_seen_segments_per_pathogen"], seg_len,
                                  err_fixed(0.05), p, rng, True)
        Xl.append(X)
        yl.append(np.full(ts["t1_seen_segments_per_pathogen"],
                          bact[slug], dtype=np.int16))
        ml.append(meta)
    dkeys = key_by_role(genome_meta, split, dec_slugs, ["test_unseen", "train"])
    dkeys2 = key_by_role(genome_meta, split, dec_slugs, ["test_unseen", "train"])
    X, meta = sample_segments(genomes, dkeys, np.ones(len(dkeys)),
                              ts["t1_decoy_segments"], seg_len,
                              err_fixed(0.05), p, rng, True)
    Xl.append(X)
    yl.append(np.full(ts["t1_decoy_segments"], dec_id, dtype=np.int16))
    ml.append(meta)
    save_npz("t1_seen_bact.npz", np.concatenate(Xl), np.concatenate(yl),
             {k: np.concatenate([m[k] for m in ml]) for k in ml[0]})

    # ---- T2 unseen-strain test ----
    rng = np.random.default_rng(422)
    Xl, yl, ml = [], [], []
    unseen_species = []
    for slug in bact:
        keys = key_by_role(genome_meta, split, [slug], ["test_unseen"])
        if not keys:
            keys = key_by_role(genome_meta, split, [slug], ["val_unseen"])
        if not keys:
            log.warning("no unseen-strain genome for %s -> excluded from T2", slug)
            continue
        unseen_species.append(slug)
        X, meta = sample_segments(genomes, keys, np.ones(len(keys)),
                                  ts["t2_unseen_segments_per_pathogen"], seg_len,
                                  err_fixed(0.05), p, rng, True)
        Xl.append(X)
        yl.append(np.full(ts["t2_unseen_segments_per_pathogen"],
                          bact[slug], dtype=np.int16))
        ml.append(meta)
    save_json({"t2_species_with_true_unseen_strain": unseen_species},
              RESULTS_DIR / "t2_species_coverage.json")
    save_npz("t2_unseen_bact.npz", np.concatenate(Xl), np.concatenate(yl),
             {k: np.concatenate([m[k] for m in ml]) for k in ml[0]})

    # ---- T3 full reads ----
    rng = np.random.default_rng(423)
    seqs, offs, srcs, labels, errs = [], [], [], [], []
    running = 0
    for slug in bact:
        keys = key_by_role(genome_meta, split, [slug],
                           ["test_unseen", "val_unseen", "train"])
        f, o, s = sample_reads(genomes, keys, ts["t3_reads_per_pathogen"], p,
                               0.05, rng)
        seqs.append(f)
        offs.append(o[1:] + running)
        running += o[-1]
        srcs.append(s)
        labels.append(np.full(ts["t3_reads_per_pathogen"], bact[slug],
                              dtype=np.int16))
        errs.append(np.full(ts["t3_reads_per_pathogen"], 0.05, dtype=np.float32))
    dkeys = key_by_role(genome_meta, split, dec_slugs, ["test_unseen", "train"])
    f, o, s = sample_reads(genomes, dkeys, ts["t3_decoy_reads"], p, 0.05, rng)
    seqs.append(f)
    offs.append(o[1:] + running)
    srcs.append(s)
    labels.append(np.full(ts["t3_decoy_reads"], dec_id, dtype=np.int16))
    errs.append(np.full(ts["t3_decoy_reads"], 0.05, dtype=np.float32))
    np.savez_compressed(READS_DIR / "t3_reads_bact.npz", seq=np.concatenate(seqs),
                        read_offset=np.concatenate(offs).astype(np.int64),
                        src=np.concatenate(srcs), label=np.concatenate(labels),
                        err=np.concatenate(errs))
    log.info("saved t3_reads_bact.npz total_bp=%d reads=%d",
             sum(len(x) for x in seqs), len(np.concatenate(labels)))

    # ---- T4 pond mixture (streaming) ----
    # Reads are emitted one by one in a random order to mimic a flow cell:
    # 95% background organisms, 5% pathogen reads.
    rng = np.random.default_rng(424)
    nd, npath = ts["t4_pond_decoy_reads"], ts["t4_pond_pathogen_reads"]
    reads = []  # list of (seq_array, label, src)

    def collect(group_keys, n_reads, label, err):
        f, o, s = sample_reads(genomes, group_keys, n_reads, p, err, rng)
        bounds = np.concatenate([[0], o])
        for i in range(n_reads):
            reads.append((f[bounds[i]:bounds[i + 1]], label, int(s[i])))

    collect(dkeys, nd, dec_id, 0.05)
    per_species = npath // len(bact)
    for slug in bact:
        keys = key_by_role(genome_meta, split, [slug],
                           ["test_unseen", "val_unseen", "train"])
        collect(keys, per_species, bact[slug], 0.05)
    order = rng.permutation(len(reads))
    flat, off, lab, src = [], [0], [], []
    for i in order:
        seg, lb, sc = reads[i]
        flat.append(seg)
        off.append(off[-1] + len(seg))
        lab.append(lb)
        src.append(sc)
    np.savez_compressed(READS_DIR / "t4_pond_mix.npz",
                        seq=np.concatenate(flat),
                        read_offset=np.array(off[1:], dtype=np.int64),
                        label=np.array(lab, dtype=np.int16),
                        src=np.array(src, dtype=np.int32))
    log.info("saved t4_pond_mix.npz reads=%d (decoy=%d, pathogen=%d)",
             len(lab), nd, npath)

    # ---- T5 robustness grid ----
    # For each (error rate x read length): sample reads of length L, cut into
    # 1kb non-overlapping windows (the classification unit), keep read_id so
    # evaluation can aggregate window probabilities per read. Reads shorter
    # than 1 kb become one N-padded window (same rule as evaluation-time
    # tail handling), so the 250/500 bp columns are actually populated.
    rng = np.random.default_rng(425)
    Xl, yl, el, ll, rid = [], [], [], [], []
    all_keys = list(bact.keys())  # slug order -> classes
    key_lists = [key_by_role(genome_meta, split, [s], ["train"])
                 for s in bact] + [dkeys2]
    n_reads_cell = ts["t5_reads_per_cell"]
    for err in p["test_sets"]["t5_error_grid"]:
        for L in p["test_sets"]["t5_length_grid"]:
            for ci, keys in enumerate(key_lists):
                cls = bact[all_keys[ci]] if ci < len(bact) else dec_id
                p_local = dict(p)
                p_local["read_length_lognormal"] = {
                    "median_bp": L, "sigma": 0.0, "min_bp": L, "max_bp": L}
                f, o, s = sample_reads(genomes, keys, n_reads_cell, p_local,
                                       err, rng)
                bounds = np.concatenate([[0], o])
                for r in range(n_reads_cell):
                    a, b2 = int(bounds[r]), int(bounds[r + 1])
                    made = 0
                    for w0 in range(a, b2 - seg_len + 1, seg_len):
                        Xl.append(f[w0:w0 + seg_len])
                        yl.append(cls)
                        el.append(err)
                        ll.append(L)
                        rid.append(r)
                        made += 1
                    tail = (b2 - a) - made * seg_len
                    if tail >= 250:
                        win = np.full(seg_len, 4, dtype=np.uint8)
                        start_tail = a + made * seg_len
                        win[:b2 - start_tail] = f[start_tail:b2]
                        Xl.append(win)
                        yl.append(cls)
                        el.append(err)
                        ll.append(L)
                        rid.append(r)
    np.savez_compressed(READS_DIR / "t5_robustness.npz", seq=np.stack(Xl),
                        label=np.array(yl, dtype=np.int16),
                        err=np.array(el, dtype=np.float32),
                        length=np.array(ll, dtype=np.int32),
                        read_id=np.array(rid, dtype=np.int32))
    log.info("saved t5_robustness.npz windows=%d", len(yl))

    # ---- T6 virus test ----
    rng = np.random.default_rng(426)
    Xl, yl, ml = [], [], []
    for slug in virus:
        keys = key_by_role(genome_meta, split, [slug], ["train", "val", "test_unseen", "val_unseen"])
        if not keys:
            log.warning("no virus genome for %s", slug)
            continue
        X, meta = sample_segments(genomes, keys, np.ones(len(keys)),
                                  ts["t6_virus_segments_per_species"], seg_len,
                                  err_fixed(0.05), p, rng, True)
        Xl.append(X)
        yl.append(np.full(ts["t6_virus_segments_per_species"], virus[slug],
                          dtype=np.int16))
        ml.append(meta)
    X, meta = sample_segments(genomes, dkeys2, np.ones(len(dkeys2)),
                              ts["t6_virus_decoy_segments"], seg_len,
                              err_fixed(0.05), p, rng, True)
    Xl.append(X)
    yl.append(np.full(ts["t6_virus_decoy_segments"], vdec_id, dtype=np.int16))
    ml.append(meta)
    save_npz("t6_virus_test.npz", np.concatenate(Xl), np.concatenate(yl),
             {k: np.concatenate([m[k] for m in ml]) for k in ml[0]})

    append_execution_log("数据集构建 | t0-t6 全部落盘（见 $AQUAPATHID_DATA/reads/）| 日志 logs/01_*.log")
    log.info("ALL TEST SETS DONE")


if __name__ == "__main__":
    main()
