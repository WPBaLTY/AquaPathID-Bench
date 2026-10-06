"""Step 03 — Train AquaPathNet on simulated ONT segments (CPU).

Training data is generated on the fly from cached genomes (infinite variety,
no fixed train array): balanced sampling across classes, per-segment error
profile drawn log-uniformly from configs/sim_params.json, random reverse-
complement augmentation.

Usage:
  python train.py --task bacteria --width 1.0 --epochs 12 --out bact
  python train.py --task virus    --width 1.0 --epochs 12 --out virus

Outputs:
  $AQUAPATHID_DATA/models/aquapathnet_<out>.pt     (checkpoint + config)
  results/train_<out>.json                          (history + final val metrics)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aqua_common import (CONFIGS_DIR, DATA_ROOT, READS_DIR, RESULTS_DIR, Timer,
                         inject_errors, kmer5_features, load_json,
                         load_species_config, pad_or_trim, reverse_complement,
                         save_json, set_all_seeds, setup_script_logger,
                         append_execution_log)
from aquapathnet import AquaPathNet, AquaPathNetV2, count_params


def build_model(arch: str, n_classes: int, width: float, task: str,
                device, log):
    """v1 = conv only; v2 = conv + 5-mer branch, trunk warm-started from v1."""
    if arch == "v1":
        return AquaPathNet(n_classes, width=width).to(device)
    model = AquaPathNetV2(n_classes, width=width).to(device)
    v1_ckpt = DATA_ROOT / "models" / f"aquapathnet_{task}.pt"
    if v1_ckpt.exists():
        ck = torch.load(v1_ckpt, map_location="cpu")
        # v1 keys (stem.*, blocks.*) map onto the v2 conv trunk (conv.*);
        # the v1 classification head is dropped (v2 has its own head)
        sd = {f"conv.{k}": v for k, v in ck["state_dict"].items()
              if not k.startswith("head")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info("warm-started v2 conv trunk from %s (%d missing, %d unexpected)",
                 v1_ckpt.name, len(missing), len(unexpected))
    return model


def forward_model(model, arch: str, X: np.ndarray, device):
    x = seq_to_input(X).to(device)
    if arch == "v1":
        return model(x)
    xk = torch.from_numpy(kmer5_features(X)).to(device)
    return model(x, xk)

MODELS_DIR = DATA_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def seq_to_input(seq: np.ndarray) -> torch.Tensor:
    """(B, L) uint8 codes -> (B, 5, L) float one-hot."""
    t = torch.from_numpy(seq.astype(np.int64))
    oh = F.one_hot(t, num_classes=5)
    return oh.permute(0, 2, 1).float()


class GenBankSampler:
    """On-the-fly balanced segment sampler from cached genomes."""

    def __init__(self, genomes: dict, split: dict, genome_meta: list[dict],
                 cfg_species: dict, task: str, p_sim: dict,
                 class_weights: dict[str, float]):
        self.genomes = genomes
        self.p = p_sim
        self.rng = np.random.default_rng(SEED)
        self.seg_len = p_sim["segment_len"]
        lo, hi = p_sim["train_total_error_range"]
        self.err_lo, self.err_hi = lo, hi
        self.rc = p_sim["reverse_comp_augmentation"]
        self.keys: list[list[str]] = []
        self.class_ids: list[int] = []
        label_map = load_json(READS_DIR / "label_map.json")
        classes = label_map["bacteria" if task == "bacteria" else "virus"]
        dec_id = (label_map["bacteria_decoy_id"] if task == "bacteria"
                  else label_map["virus_decoy_id"])
        dec_slugs = [d["slug"] for d in cfg_species["decoys"]]
        for slug, cid in classes.items():
            keys = [m["key"] for m in genome_meta
                    if split[m["key"]]["slug"] == slug
                    and split[m["key"]]["role"].startswith("train")]
            if keys:
                self.keys.append(keys)
                self.class_ids.append(cid)
        dkeys = [m["key"] for m in genome_meta
                 if split[m["key"]]["slug"] in dec_slugs
                 and split[m["key"]]["role"].startswith("train")]
        if dkeys:
            self.keys.append(dkeys)
            self.class_ids.append(dec_id)
        self.class_weights = np.array(
            [class_weights.get(str(cid), 1.0) for cid in self.class_ids],
            dtype=np.float64)
        self.class_weights /= self.class_weights.sum()

    def sample_class(self, ci: int, n: int):
        keys = self.keys[ci]
        chosen = self.rng.integers(0, len(keys), size=n)
        out = np.empty((n, self.seg_len), dtype=np.uint8)
        for i in range(n):
            arr = self.genomes[keys[chosen[i]]]
            # short-window augmentation: with prob 0.15 draw a shorter window
            # and right-pad with N, so short reads are in-distribution
            if self.rng.random() < 0.15:
                short_len = int(self.rng.integers(250, self.seg_len))
            else:
                short_len = self.seg_len
            start = int(self.rng.integers(0, max(1, len(arr) - short_len)))
            seg = arr[start:start + short_len].copy()
            if len(seg) < self.seg_len:
                seg = np.pad(seg, (0, self.seg_len - len(seg)),
                             constant_values=4)
            err = float(np.exp(self.rng.uniform(np.log(self.err_lo),
                                                np.log(self.err_hi))))
            seg = inject_errors(seg, err, self.p, self.rng)
            seg = pad_or_trim(seg, self.seg_len)
            if self.rc and self.rng.random() < 0.5:
                seg = reverse_complement(seg)
            out[i] = seg
        return out

    def batch(self, bs: int):
        """Return (X uint8 (bs,L), y int64 (bs)) with class-balanced sampling."""
        counts = self.rng.multinomial(bs, self.class_weights)
        xs, ys = [], []
        for ci, n in enumerate(counts):
            if n == 0:
                continue
            xs.append(self.sample_class(ci, int(n)))
            ys.append(np.full(int(n), self.class_ids[ci], dtype=np.int64))
        X = np.concatenate(xs)
        y = np.concatenate(ys)
        perm = self.rng.permutation(len(y))
        return X[perm], y[perm]


def evaluate_val(model, arch, val_npz, device) -> dict:
    from sklearn.metrics import f1_score
    seq, y = val_npz["seq"], val_npz["label"].astype(np.int64)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(y), 512):
            logits = forward_model(model, arch, seq[i:i + 512], device)
            preds.append(logits.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    return {
        "acc": float((preds == y).mean()),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "n": int(len(y)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["bacteria", "virus"], default="bacteria")
    ap.add_argument("--arch", choices=["v1", "v2"], default="v1")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--segments-per-class", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_name = args.out or args.task
    log = setup_script_logger(f"train_{out_name}")
    set_all_seeds(SEED)
    torch.set_num_threads(8)
    device = torch.device("cpu")

    p_sim = load_json(CONFIGS_DIR / "sim_params.json")
    cfg_species = load_species_config()
    genomes = {k: v for k, v in np.load(READS_DIR / "genome_cache.npz").items()}
    split = load_json(READS_DIR / "split_map.json")
    genome_meta = load_json(READS_DIR / "genome_meta.json")
    label_map = load_json(READS_DIR / "label_map.json")

    classes = label_map["bacteria"] if args.task == "bacteria" else label_map["virus"]
    n_classes = len(classes) + 1  # + background

    # background class oversampled x3: matches relative abundance of a pond
    # metagenome better than 1:1 while keeping all classes learnable
    cw = {str(i): 1.0 for i in range(n_classes)}
    cw[str(n_classes - 1)] = 3.0
    sampler = GenBankSampler(genomes, split, genome_meta, cfg_species,
                             args.task, p_sim, cw)

    model = build_model(args.arch, n_classes, args.width, args.task,
                        device, log)
    n_params = count_params(model)
    log.info("arch=%s model params: %d (%.2f MB fp32)", args.arch, n_params,
             n_params * 4 / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = args.segments_per_class * n_classes // args.batch
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.15,
        div_factor=10, final_div_factor=20)

    val = np.load(READS_DIR / f"t0_val_{'bact' if args.task == 'bacteria' else 'virus'}.npz")
    history, best_f1, t_start = [], -1.0, time.perf_counter()
    ckpt_name = (f"aquapathnet_{out_name}.pt" if args.arch == "v1"
                 else f"aquapathnet2_{out_name}.pt")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_n, t0 = 0.0, 0, time.perf_counter()
        for _ in range(steps_per_epoch):
            X, y = sampler.batch(args.batch)
            yt = torch.from_numpy(y).to(device)
            logits = forward_model(model, args.arch, X, device)
            loss = F.cross_entropy(logits, yt, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ep_loss += float(loss.detach()) * len(y)
            ep_n += len(y)
            step += 1
        m = evaluate_val(model, args.arch, val, device)
        history.append({"epoch": epoch + 1, "loss": ep_loss / ep_n,
                        "val_acc": m["acc"], "val_macro_f1": m["macro_f1"],
                        "sec": round(time.perf_counter() - t0, 1)})
        log.info("epoch %2d/%d loss=%.4f val_acc=%.4f val_mF1=%.4f (%.0fs)",
                 epoch + 1, args.epochs, ep_loss / ep_n, m["acc"],
                 m["macro_f1"], history[-1]["sec"])
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save({"state_dict": model.state_dict(),
                        "n_classes": n_classes, "width": args.width,
                        "task": args.task, "arch": args.arch,
                        "label_map_subset": {**classes, "BACKGROUND": n_classes - 1}},
                       MODELS_DIR / ckpt_name)
            log.info("  -> saved new best (mF1=%.4f)", best_f1)

    final = evaluate_val(model, args.arch, val, device)
    save = {"task": args.task, "arch": args.arch, "width": args.width,
            "params": n_params, "epochs": args.epochs, "history": history,
            "best_val_macro_f1": best_f1, "final_val_macro_f1": final["macro_f1"],
            "train_seconds": round(time.perf_counter() - t_start, 1),
            "seed": SEED, "segments_per_class": args.segments_per_class}
    save_json(save, RESULTS_DIR / f"train_{out_name}.json")
    append_execution_log(
        f"训练 {out_name} arch={args.arch} | width={args.width} 参数={n_params} "
        f"best_val_mF1={best_f1:.4f} 用时={save['train_seconds']}s | "
        f"results/train_{out_name}.json")
    log.info("TRAIN DONE best_val_macro_f1=%.4f", best_f1)


if __name__ == "__main__":
    main()
