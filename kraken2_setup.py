# -*- coding: utf-8 -*-
"""Kraken2 official-baseline staging (everything to prepare before the WSL run):
1) export T1/T2/T3/T4 query FASTA from the npz files
2) download the NCBI taxdump (names/nodes)
3) build kraken2_db: taxonomy + seqid2taxid.map + library/addition.fna
   (the same 104 training genomes as every other baseline, strictly one library)
4) download the Kraken2 source (from GitHub)
5) pre-write the one-shot WSL script setup_and_run.sh
"""
import os
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

DATA = Path(os.environ.get("AQUAPATHID_DATA", "./AquaPathID_data"))
READS = DATA / "reads"
FASTA_DIR = DATA / "fasta"
WORK = DATA / "kraken2_work"
DB = DATA / "kraken2_db"
SRC = DATA / "kraken2_src"
SETS = ["t1_seen_bact", "t2_unseen_bact", "t3_reads_bact", "t4_pond_mix"]

WORK.mkdir(exist_ok=True)
DB.mkdir(exist_ok=True)
(DB / "taxonomy").mkdir(exist_ok=True)
(DB / "library").mkdir(exist_ok=True)
SRC.mkdir(exist_ok=True)

split_map = json.loads((READS / "split_map.json").read_text(encoding="utf-8"))
meta = json.loads((READS / "genome_meta.json").read_text(encoding="utf-8"))
manifest = json.loads(
    (Path(__file__).resolve().parent / "results"
     / "genome_manifest.json").read_text(encoding="utf-8"))
label_map = json.loads((READS / "label_map.json").read_text(encoding="utf-8"))
target_ids = label_map["bacteria"]          # slug -> class id (0..23)
decoy_id = label_map["bacteria_decoy_id"]   # 24 = background

# the virus manifest has no taxids and its name spellings differ from
# NCBI, so the real NCBI leaf taxids are hard-coded here
TAXID_OVERRIDES = {"nnv": "43763",      # Redspotted grouper nervous necrosis virus
                   "svcv": "696863",    # Spring viraemia of carp virus
                   "wssv": "92652"}     # Shrimp white spot syndrome virus

# ---- 1) export query FASTA ----
for name in SETS:
    z = np.load(READS / f"{name}.npz", allow_pickle=True)
    keys = list(z.keys())
    seqs = z["seq"]
    out = WORK / f"{name}.fna"
    code = "ACGTN"
    n_written = 0
    with open(out, "w", newline="\n") as fh:
        if seqs.ndim == 2:   # fixed-length windows
            for i, s in enumerate(seqs):
                fh.write(f">{name}_r{i}\n")
                fh.write("".join(code[int(b)] for b in s) + "\n")
                n_written += 1
        else:                # flat array + read_offset (variable-length reads)
            # read_offset in the npz stores each read's END position (the
            # first read starts at 0), matching the evaluate.py convention
            # starts = concat([[0], off[:-1]])
            ends = z["read_offset"].astype(np.int64)
            starts = np.concatenate([[0], ends[:-1]])
            total = len(seqs)
            for i in range(len(ends)):
                a, b = int(starts[i]), int(ends[i])
                fh.write(f">{name}_r{i}\n")
                fh.write("".join(code[int(v)] for v in seqs[a:b]) + "\n")
                n_written += 1
    print(f"{name}: keys={keys} n={n_written} -> {out.name} "
          f"({out.stat().st_size/1e6:.1f} MB)")

# ---- 2) taxdump ----
taxdump_tgz = WORK / "taxdump.tar.gz"
if not (DB / "taxonomy" / "nodes.dmp").exists():
    if not taxdump_tgz.exists():
        print("downloading taxdump ...")
        urllib.request.urlretrieve(
            "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz",
            taxdump_tgz)
    with tarfile.open(taxdump_tgz) as tf:
        for member in ("names.dmp", "nodes.dmp"):
            fh = tf.extractfile(member)
            (DB / "taxonomy" / member).write_bytes(fh.read())
    print("taxdump extracted")
else:
    print("taxdump already staged")

# ---- 3) species -> taxid ----
name2taxid = {}
node_set = set()
with open(DB / "taxonomy" / "names.dmp", encoding="utf-8") as fh:
    for line in fh:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[3] == "scientific name":
            name2taxid[parts[1].lower()] = parts[0]
with open(DB / "taxonomy" / "nodes.dmp", encoding="utf-8") as fh:
    for line in fh:
        p = line.split("|", 1)[0].strip()
        node_set.add(p)

def find_taxid(slug, acc):
    if slug in TAXID_OVERRIDES:
        return TAXID_OVERRIDES[slug], "override"
    rec = man_by_key.get((slug, acc)) or man_by_slug.get(slug)
    nm = (rec or {}).get("species", "")
    # prefer the current taxid resolved from the scientific name (the
# manifest value may be a stale/retired id)
    if nm.lower() in name2taxid and name2taxid[nm.lower()] in node_set:
        return name2taxid[nm.lower()], "names.dmp"
    if rec and rec.get("taxid") and str(rec["taxid"]) in node_set:
        return str(rec["taxid"]), "manifest"
    return None, nm

# manifest index: the genomes list, with slug/assembly/taxid/species fields
man_by_key = {}
man_by_slug = {}
for rec in manifest["genomes"]:
    if not isinstance(rec, dict):
        continue
    if rec.get("slug"):
        man_by_slug.setdefault(rec["slug"], rec)
    if rec.get("slug") and rec.get("assembly"):
        man_by_key[(rec["slug"], rec["assembly"])] = rec

n_train = n_mapped = 0
missing = []
with open(DB / "seqid2taxid.map", "w", newline="\n") as out:
    for key, role_rec in split_map.items():
        if role_rec.get("role") != "train":
            continue
        slug, acc = key.split("#", 1)
        n_train += 1
        taxid, how = find_taxid(slug, acc)
        if taxid is None:
            missing.append(slug)
            continue
        out.write(f"{slug}|{acc}\t{taxid}\n")
        n_mapped += 1
print(f"train genomes={n_train}, mapped={n_mapped}, missing={sorted(set(missing))}")

# stage library/addition.fna (equivalent to kraken2-build --add-to-library)
addition = DB / "library" / "addition.fna"
if not addition.exists():
    shutil_copy = open(FASTA_DIR / "ref_bacteria.fna", "rb").read()
    addition.write_bytes(shutil_copy)
    print(f"library/addition.fna staged "
          f"({addition.stat().st_size/1e6:.0f} MB)")

# ---- 4) Kraken2 source ----
ver_file = SRC / "kraken2-master.tar.gz"
if not ver_file.exists():
    print("downloading kraken2 source ...")
    try:
        urllib.request.urlretrieve(
            "https://codeload.github.com/DerrickWood/kraken2/tar.gz/refs/heads/master",
            ver_file)
    except Exception as exc:
        print("urllib failed:", exc)
        import subprocess
        subprocess.run(["curl.exe" if os.name == "nt" else "curl", "-L", "-o", str(ver_file),
                        "https://codeload.github.com/DerrickWood/kraken2/"
                        "tar.gz/refs/heads/master"], check=True)
    print("kraken2 source:", ver_file.stat().st_size / 1e6, "MB")

# ---- 5) one-shot WSL script ----
# convert the data root to a WSL mount path (D:\data -> /mnt/d/data;
# POSIX paths pass through unchanged)
if len(DATA.drive) == 2 and DATA.drive[1] == ":":
    wsl_data = "/mnt/" + DATA.drive[0].lower() + DATA.as_posix()[2:]
else:
    wsl_data = DATA.as_posix()
sh = r"""#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
  sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources
fi
[ -f /etc/apt/sources.list ] && sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list || true
apt-get update -y
apt-get install -y g++ make zlib1g-dev
cd __WSL_DATA__/kraken2_src
[ -d kraken2-master ] || tar xzf kraken2-master.tar.gz
cd kraken2-master
make -j16
DB=__WSL_DATA__/kraken2_db
./kraken2-build --db $DB --build --threads 16 --no-masking 2>&1 | tee __WSL_DATA__/kraken2_work/build.log
./kraken2-inspect --db $DB | head -5 > __WSL_DATA__/kraken2_work/inspect.txt || true
for s in t1_seen_bact t2_unseen_bact t3_reads_bact t4_pond_mix; do
  ./kraken2 --db $DB --threads 16 \
    --output __WSL_DATA__/kraken2_work/${s}.k2.out \
    __WSL_DATA__/kraken2_work/${s}.fna
done
echo KRAKEN2_ALL_DONE
""".replace("__WSL_DATA__", wsl_data)
(WORK / "setup_and_run.sh").write_text(sh, newline="\n")
print("WSL script ready:", WORK / "setup_and_run.sh")
print("STAGING_COMPLETE")
