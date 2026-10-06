"""Step 07 — External validation on public ONT runs (ENA/SRA).

Pipeline:
  1. Search ENA portal for MinION/GridION WGS runs of target species (50-600 MB).
  2. Download the FASTQ (https), stream-parse, subsample up to max_reads.
  3. Ground truth: megablast (NCBI BLAST+) against the training reference;
     keep reads with coverage>=0.5 and identity>=0.85, species of best HSP.
  4. Classify the same reads with AquaPathNet; compare vs BLAST truth.

Usage:
  python 07_real_data.py --search            # find and list candidate runs
  python 07_real_data.py --run <run_acc> --slug <species_slug> --taxid 670
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import urllib.request
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aqua_common import (DATA_ROOT, READS_DIR, RESULTS_DIR, seq_to_array,
                         save_json, setup_script_logger, append_execution_log)
from aqua_common import load_json, load_species_config

import importlib
_seq_to_input = import_module("train").seq_to_input

ONT_DIR = DATA_ROOT / "ont_real"
BLAST_ROOT = (Path(os.environ.get("AQUAPATHID_TOOLS", "./tools"))
              / "ncbi-blast-2.17.0+" / "bin")
BLAST_EXE = "blastn.exe" if os.name == "nt" else "blastn"
ENA = "https://www.ebi.ac.uk/ena/portal/api/search"


def ena_search(cfg: dict, taxid: int, log) -> list[dict]:
    query = f'tax_tree({taxid}) AND instrument_platform=OXFORD_NANOPORE'
    url = (f"{ENA}?result=read_run&query={urllib.request.quote(query)}"
           "&fields=run_accession,scientific_name,instrument_model,fastq_bytes,"
           "fastq_ftp,read_count&format=json&limit=100")
    req = urllib.request.Request(url, headers={"User-Agent": "AquaPathID/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    rows = data if isinstance(data, list) else []
    ok = []
    for row in rows:
        try:
            nbytes = int(row.get("fastq_bytes", "0").split(";")[0])
        except ValueError:
            continue
        if cfg["fastq_bytes_min"] <= nbytes <= cfg["fastq_bytes_max"]:
            ok.append(row)
    log.info("taxid %d: %d runs total, %d in size window", taxid, len(rows), len(ok))
    return ok


def download_fastq(ftp_url: str, dest: Path, log) -> Path:
    if dest.exists() and dest.stat().st_size > 1e6:
        return dest
    url = ftp_url
    if url.startswith("ftp://"):
        url = "https://" + url[len("ftp://"):]
    elif not url.startswith("http"):
        url = "https://" + url
    tmp = dest.with_suffix(".tmp")
    log.info("downloading %s -> %s", url, dest.name)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def fastq_to_records(fq: Path, max_reads: int):
    """Stream a fastq.gz, yield (header, seq_code_array). Subsample evenly."""
    # count total reads first (cheap-ish pass on compressed file? do single pass
    # storing headers only, then second pass decoding chosen reads)
    headers = []
    with gzip.open(fq, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i % 4 == 0:
                headers.append(line.strip()[1:])
    n = len(headers)
    rng = np.random.default_rng(11)
    if n > max_reads:
        pick = np.sort(rng.choice(n, max_reads, replace=False))
    else:
        pick = np.arange(n)
    pick_set = set(pick.tolist())
    out = {}
    idx = -1
    with gzip.open(fq, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            m = i % 4
            if m == 0:
                idx += 1
                cur = None
                if idx in pick_set:
                    cur = idx
            elif cur is not None and m == 1:
                out[cur] = (headers[idx], seq_to_array(line.strip().upper()))
    return [out[i] for i in sorted(out.keys())]


def blast_truth(records, ref_fasta: Path, cfg: dict, log, out_tsv: Path):
    q = out_tsv.with_suffix(".query.fna")
    with open(q, "w", encoding="utf-8", newline="\n") as f:
        for i, (hdr, seq) in enumerate(records):
            s = "".join("ACGTN"[int(b)] for b in seq)
            f.write(f">r{i}\n{s}\n")
    if not out_tsv.exists():
        t0 = time.perf_counter()
        with open(out_tsv, "w") as fo:
            subprocess.run(
                [str(BLAST_ROOT / BLAST_EXE), "-task", "megablast",
                 "-query", str(q), "-subject", str(ref_fasta), "-outfmt",
                 "6 qseqid sseqid qlen qstart qend nident length",
                 "-num_threads", "8", "-evalue", "1e-10", "-dust", "no"],
                stdout=fo, check=True)
        bt = time.perf_counter() - t0
        log.info("BLAST truth done in %.0fs", bt)
    else:
        bt = None
    classes = load_json(READS_DIR / "label_map.json")["bacteria"]
    # aggregate HSPs per (query, subject); assign the subject with the largest
    # merged query coverage (single-HSP coverage rejects error-containing reads)
    hsp, qlen_of = {}, {}
    with open(out_tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            qid, sid, qlen, qs, qe, nident, length = parts
            qlen_of[qid] = int(qlen)
            hsp.setdefault(qid, {}).setdefault(sid, []).append(
                (int(qs), int(qe), int(nident), int(length)))
    truth = {}
    ident_of = {}
    cov_of = {}
    for i in range(len(records)):
        qid = f"r{i}"
        if qid not in hsp:
            truth[i] = None
            ident_of[i] = 0.0
            cov_of[i] = 0.0
            continue
        best_sid, best_cov, best_ident, best_merged = "", 0.0, 0.0, -1
        for sid, hsps in hsp[qid].items():
            merged, last_end = 0, -1
            ident_sum = len_sum = 0
            for qs, qe, nident, length in sorted(hsps):
                ident_sum += nident
                len_sum += length
                merged += max(0, qe - max(qs, last_end + 1))
                last_end = max(last_end, qe)
            cov = merged / max(1, qlen_of[qid])
            ident = ident_sum / max(1, len_sum)
            if merged > best_merged:
                best_sid, best_cov, best_ident, best_merged = (
                    sid, cov, ident, merged)
        ident_of[i] = best_ident
        cov_of[i] = best_cov
        if best_cov < cfg["blast_cov_min"] or best_ident < cfg["blast_ident_min"]:
            truth[i] = None
        else:
            slug = best_sid.split("|")[0]
            truth[i] = classes.get(slug, None)
    return truth, ident_of, cov_of, bt, q


def mappy_truth(records, ref_fasta: Path, cfg: dict, log):
    """Truth via minimap2 (map-ont) — fast on divergent metagenomic reads.
    Same acceptance thresholds as the BLAST path."""
    import mappy
    classes = load_json(READS_DIR / "label_map.json")["bacteria"]
    t0 = time.perf_counter()
    aligner = mappy.Aligner(fn_idx_in=str(ref_fasta), preset="map-ont")
    log.info("mappy index built in %.0fs", time.perf_counter() - t0)
    truth, ident_of, cov_of = {}, {}, {}
    for i, (hdr, seq) in enumerate(records):
        s = "".join("ACGTN"[int(b)] for b in seq)
        per_sid = {}
        for hit in aligner.map(s):
            per_sid.setdefault(hit.ctg, []).append(hit)
        best_sid, best_cov, best_ident, best_merged = "", 0.0, 0.0, -1
        for sid, hits in per_sid.items():
            merged, last_end = 0, -1
            ident_sum = len_sum = 0
            for h in sorted(hits, key=lambda x: x.q_st):
                ident_sum += h.mlen
                len_sum += h.blen
                merged += max(0, h.q_en - max(h.q_st, last_end + 1))
                last_end = max(last_end, h.q_en)
            cov = merged / max(1, len(s))
            ident = ident_sum / max(1, len_sum)
            if merged > best_merged:
                best_sid, best_cov, best_ident, best_merged = (
                    sid, cov, ident, merged)
        ident_of[i], cov_of[i] = best_ident, best_cov
        slug = best_sid.split("|")[0] if best_sid else ""
        if not slug or best_cov < cfg["blast_cov_min"] \
                or best_ident < cfg["blast_ident_min"]:
            truth[i] = None
        else:
            truth[i] = classes.get(slug, None)
    return truth, ident_of, cov_of, None, None


def model_classify(records, log):
    import torch
    import torch.nn.functional as F
    from aqua_common import kmer5_features
    from aquapathnet import AquaPathNet, AquaPathNetV2
    p2 = DATA_ROOT / "models" / "aquapathnet2_bacteria.pt"
    path = p2 if p2.exists() else DATA_ROOT / "models" / "aquapathnet_bacteria.pt"
    ckpt = torch.load(path, map_location="cpu")
    arch = ckpt.get("arch", "v1")
    if arch == "v1":
        model = AquaPathNet(ckpt["n_classes"], width=ckpt["width"])
    else:
        model = AquaPathNetV2(ckpt["n_classes"], width=ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    torch.set_num_threads(8)
    inv = {v: k for k, v in ckpt["label_map_subset"].items()}
    seg_len = 1000
    min_tail = 250
    t0 = time.perf_counter()
    preds = []
    with torch.no_grad():
        for hdr, seq in records:
            wins = []
            n_full = len(seq) // seg_len
            for w in range(n_full):
                wins.append(seq[w * seg_len:(w + 1) * seg_len])
            tail = len(seq) - n_full * seg_len
            if tail >= min_tail:
                win = np.full(seg_len, 4, dtype=np.uint8)
                win[:tail] = seq[n_full * seg_len:]
                wins.append(win)
            if not wins:
                preds.append(-1)
                continue
            X = np.stack(wins)
            probs = []
            for i in range(0, len(X), 256):
                x = _seq_to_input(X[i:i + 256])
                if arch == "v2":
                    xk = torch.from_numpy(kmer5_features(X[i:i + 256]))
                    probs.append(F.softmax(model(x, xk), dim=1).cpu().numpy())
                else:
                    probs.append(F.softmax(model(x), dim=1).cpu().numpy())
            p = np.concatenate(probs).mean(0)
            preds.append(int(p.argmax()))
    dt = time.perf_counter() - t0
    log.info("model (%s) classified %d reads in %.1fs (%.1f reads/s)",
             arch, len(records), dt, len(records) / dt)
    return preds, dt, inv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--slug", default=None)
    ap.add_argument("--expect", default=None,
                    help="expected slug (for reporting)")
    ap.add_argument("--max-reads", type=int, default=None)
    ap.add_argument("--ftp", default=None,
                    help="direct fastq https/ftp URL (overrides candidates)")
    ap.add_argument("--truth-tool", default="blast", choices=["blast", "mappy"],
                    help="truth assignment engine (mappy = minimap2 map-ont)")
    ap.add_argument("--skip-model", action="store_true",
                    help="only build the BLAST truth TSV (model added later)")
    args = ap.parse_args()
    log = setup_script_logger("07_real_data")
    cfg = load_json(Path(__file__).resolve().parents[1] / "configs" /
                    "real_data.json")
    ONT_DIR.mkdir(parents=True, exist_ok=True)

    if args.search:
        found = {}
        for item in cfg["target_species"]:
            rows = ena_search(cfg, item["taxid"], log)
            if rows:
                rows.sort(key=lambda r: int(r.get("fastq_bytes", "0")
                                            .split(";")[0]))
                found[item["slug"]] = rows[0]
                log.info("candidate for %s: %s (%s, %s bytes)",
                         item["slug"], rows[0]["run_accession"],
                         rows[0]["scientific_name"], rows[0]["fastq_bytes"])
        save_json(found, RESULTS_DIR / "real_data_candidates.json")
        log.info("candidates saved: %d species", len(found))
        return

    if not (args.run and args.slug):
        log.error("--run and --slug required (or --search)")
        return
    max_reads = args.max_reads or cfg["max_reads"]
    meta = None
    cands = RESULTS_DIR / "real_data_candidates.json"
    if cands.exists():
        meta = load_json(cands).get(args.slug)
    if args.ftp:
        fq_url = args.ftp
    else:
        fq_url = (meta["fastq_ftp"].split(";")[0]
                  if meta and meta.get("fastq_ftp") else None)
    if not fq_url:
        log.error("no fastq_ftp known for %s; run --search first", args.slug)
        return
    existing = sorted(ONT_DIR.glob(f"{args.run}*.fastq.gz"))
    if existing:
        dest = existing[0]
        log.info("using existing fastq %s", dest.name)
    else:
        dest = ONT_DIR / f"{args.run}.fastq.gz"
        download_fastq(fq_url, dest, log)
    records = fastq_to_records(dest, max_reads)
    log.info("loaded %d reads (%s)", len(records), dest.name)
    ref = DATA_ROOT / "fasta" / "ref_bacteria.fna"
    if not ref.exists():
        log.error("reference fasta missing; run 05_baselines first")
        return
    if args.truth_tool == "mappy":
        truth, ident_of, cov_of, bt, q = mappy_truth(
            records, ref, cfg, log)
    else:
        truth, ident_of, cov_of, bt, q = blast_truth(
            records, ref, cfg, log, ONT_DIR / f"{args.run}_blast.tsv")
    if args.skip_model:
        n_classified = sum(1 for v in truth.values() if v is not None)
        log.info("truth-only: %d/%d reads BLAST-classified; stopping "
                 "(--skip-model)", n_classified, len(records))
        return
    preds, model_s, inv = model_classify(records, log)

    n_classified = sum(1 for v in truth.values() if v is not None)
    agree = 0
    agree_genus = 0
    conf = {}
    expect = args.expect or args.slug
    for i, p in enumerate(preds):
        t = truth.get(i)
        if t is None:
            continue
        slug_pred = inv.get(p, "BACKGROUND")
        slug_true = inv.get(t, "?")
        conf[(slug_true, slug_pred)] = conf.get((slug_true, slug_pred), 0) + 1
        if p == t:
            agree += 1
        elif (slug_pred != "BACKGROUND" and slug_true != "?"
              and slug_pred.split("_")[0] == slug_true.split("_")[0]):
            agree_genus += 1
    n_model_pos = sum(1 for p in preds
                      if p >= 0 and inv.get(p, "BACKGROUND") != "BACKGROUND")
    n_model_expected = sum(1 for p in preds if inv.get(p) == expect)
    # identity-stratified agreement (mitigates shared-reference circularity:
    # high-identity strata have near-certain truth)
    strata = {"ident>=0.99": [0, 0], "0.95<=ident<0.99": [0, 0],
              "0.85<=ident<0.95": [0, 0]}
    per_read = {"truth": [], "pred": [], "ident": [], "cov": []}
    for i, p in enumerate(preds):
        t = truth.get(i)
        per_read["truth"].append(inv.get(t, "UNCLASSIFIED") if t is not None
                                 else "UNCLASSIFIED")
        per_read["pred"].append(inv.get(p, "BACKGROUND") if p >= 0
                                else "TOO_SHORT")
        per_read["ident"].append(float(ident_of.get(i, 0.0)))
        per_read["cov"].append(float(cov_of.get(i, 0.0)))
        if t is None:
            continue
        ident = ident_of.get(i, 0.0)
        key = ("ident>=0.99" if ident >= 0.99 else
               "0.95<=ident<0.99" if ident >= 0.95 else "0.85<=ident<0.95")
        strata[key][1] += 1
        if p == t:
            strata[key][0] += 1
    np.savez_compressed(ONT_DIR / f"{args.run}_perread.npz",
                        truth=np.array(per_read["truth"]),
                        pred=np.array(per_read["pred"]),
                        ident=np.array(per_read["ident"], dtype=np.float32),
                        cov=np.array(per_read["cov"], dtype=np.float32))
    res = {
        "run": args.run, "expected_species": expect, "n_reads": len(records),
        "n_blast_classified": n_classified,
        "model_agreement_with_blast_truth": (agree / max(1, n_classified)),
        "model_genus_level_agreement": (agree_genus / max(1, n_classified)),
        "agreement_by_identity": {k: (round(v[0] / v[1], 3) if v[1] else None)
                                  for k, v in strata.items()},
        "n_by_identity": {k: v[1] for k, v in strata.items()},
        "model_positive_call_rate": n_model_pos / max(1, len(preds)),
        "model_expected_species_call_rate": n_model_expected / max(1, len(preds)),
        "model_reads_per_second": len(records) / model_s,
        "blast_seconds": bt,
        "model_seconds": model_s,
        "confusion_true_vs_pred": {f"{k[0]}->{k[1]}": v
                                   for k, v in sorted(conf.items(),
                                                      key=lambda x: -x[1])[:20]},
    }
    save_json(res, RESULTS_DIR / f"real_data_{args.run}.json")
    log.info("SUMMARY %s: blast-classified=%d agreement=%.3f "
             "model=%.1f reads/s", args.run, n_classified,
             res["model_agreement_with_blast_truth"],
             res["model_reads_per_second"])
    append_execution_log(
        f"实测验证 {args.run} ({expect}) | reads={len(records)} | "
        f"agreement={res['model_agreement_with_blast_truth']:.3f} | "
        f"results/real_data_{args.run}.json")


if __name__ == "__main__":
    main()
