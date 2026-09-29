"""Fast end-to-end sanity check of data, preprocessing, models and metrics (~1-2 minutes).

Runs every stage of the pipeline on a small, patient-balanced subset of DMR-IR and checks
formats, value ranges, parity with the training datasets, weight loading, activations and
metric implementations against scikit-learn. Writes runs/pipeline_check.md.

Usage:
    python scripts/check_pipeline.py
"""

import glob
import hashlib
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn import metrics as skm

from thermal_bc.data.dmrir import list_dmrir_images, patient_id_from_path
from thermal_bc.data.thermal_dataset import ThermalDataset
from thermal_bc.evaluation import binary_scores
from thermal_bc.features import encoder_input
from thermal_bc.models.metrics import confusion_metrics, get_metrics
from thermal_bc.pipeline import (
    EMB_SIZE,
    SEG_SIZE,
    ThermalPipeline,
    embedding_input,
    load_thermogram,
    minmax,
    segmentation_input,
)

DB = Path("data/dmrir/lab_database")
SEG_WEIGHTS = Path("weights/segmentation_r2attunet.pth")
SEG_CONFIG = Path("weights/segmentation_r2attunet.json")
BACKBONE = Path("weights/vicreg_resnet50.pth")
N_PATIENTS_PER_CLASS, FRAMES_PER_PATIENT = 4, 5

results = []


def check(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append((status, name, detail))
    print(f"[{status}] {name}: {detail}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def subset():
    paths, labels = list_dmrir_images(DB / "database")
    by_patient = defaultdict(list)
    for p, y in zip(paths, labels):
        by_patient[(patient_id_from_path(p), y)].append(p)
    rng = random.Random(0)
    chosen = []
    for label in (0, 1):
        patients = sorted(k for k in by_patient if k[1] == label and not k[0].startswith("20"))
        for key in rng.sample(patients, N_PATIENTS_PER_CLASS):
            chosen += [(p, label, key[0]) for p in sorted(by_patient[key])[:FRAMES_PER_PATIENT]]
    return chosen


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = subset()
    paths = [p for p, _, _ in items]
    print(f"Subset: {len(items)} frames from {len({g for *_, g in items})} patients\n")

    # ------------------------------------------------------------------ A. data
    all_files = glob.glob(str(DB / "database" / "*" / "*"))
    shapes, non_gray = defaultdict(int), []
    for f in all_files:
        raw = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        shapes[raw.shape] += 1
        if raw.ndim == 3 and not (
            np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 1], raw[..., 2])
        ):
            diff = np.abs(raw[..., 0].astype(int) - raw[..., 2].astype(int)).max()
            non_gray.append((os.path.basename(f), diff))
    check(
        "A1 DMR-IR frames readable",
        len(all_files) == 1000,
        f"{len(all_files)} files, shapes {dict(shapes)}",
    )
    max_diff = max((d for _, d in non_gray), default=0)
    check(
        "A2 3-channel frames are grayscale (R=G=B)",
        max_diff <= 8,
        f"{len(non_gray)} RGB files, max channel difference {max_diff}",
        warn=True,
    )
    # Optional: raw DMR-IR temperature matrices, if present, must be plausible body temperatures.
    txt = sorted(glob.glob("data/dmrir/**/*.txt", recursive=True))[:3]
    if txt:
        temps = [load_thermogram(t) for t in txt]
        check(
            "A3 temperature matrices",
            all(t.shape == (480, 640) and 20 < t.min() and t.max() < 40 for t in temps),
            ", ".join(f"{t.shape} {t.min():.1f}-{t.max():.1f} C" for t in temps),
        )
    gt_paths = [DB / "labels" / Path(p).parent.name / Path(p).name for p in paths]
    gt_masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127 for p in gt_paths]
    fractions = [m.mean() for m in gt_masks]
    check(
        "A4 DMR-IR breast masks (labels/)",
        all(0.1 < f < 0.7 for f in fractions),
        f"binarised at 127, breast fraction {np.min(fractions):.2f}-{np.max(fractions):.2f}",
    )

    # ------------------------------------------------------------------ B. preprocessing parity
    x = embedding_input(load_thermogram(paths[0]))
    check(
        "B1 embedding input range/shape",
        tuple(x.shape) == (1, 3, *EMB_SIZE) and 0 <= x.min() and x.max() <= 1,
        f"{tuple(x.shape)} [{x.min():.2f}, {x.max():.2f}]",
    )
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(paths[:4]):
            np.save(Path(tmp) / f"img_{i}.npy", load_thermogram(p))
        tds = ThermalDataset(tmp, height=SEG_SIZE[0], width=SEG_SIZE[1])
        diffs = [
            (segmentation_input(np.load(Path(tmp) / tds.image_files[i]))[0] - tds[i][0])
            .abs()
            .max()
            .item()
            for i in range(4)
        ]
    check(
        "B2 segmentation input == ThermalDataset (training)",
        max(diffs) < 1e-6,
        f"max |diff| {max(diffs):.2e}, size {SEG_SIZE}",
    )
    const = np.full((50, 50), 7.0, np.float32)
    check(
        "B3 constant image does not produce NaN",
        not np.isnan(minmax(const)).any(),
        "pipeline minmax",
    )

    # ------------------------------------------------------------------ C. segmentation model
    pipe = ThermalPipeline(SEG_WEIGHTS, SEG_CONFIG, BACKBONE, device=device)
    seg = pipe.segmenter
    check(
        "C1 segmentation weights",
        not seg.training,
        f"sha256 {sha256(SEG_WEIGHTS)}, strict load ok, "
        f"{sum(p.numel() for p in seg.parameters()) / 1e6:.2f}M params, eval mode",
    )
    batch = torch.cat([segmentation_input(load_thermogram(p)) for p in paths[:8]]).to(device)
    with torch.no_grad():
        logits = seg(batch)
        single = seg(batch[:1])
    probs = logits.softmax(1)
    check(
        "C2 logits shape/finite",
        tuple(logits.shape) == (8, 3, *SEG_SIZE) and torch.isfinite(logits).all().item(),
        f"{tuple(logits.shape)}, softmax sums to {probs.sum(1).mean():.4f}",
    )
    check(
        "C3 eval-mode batch invariance",
        torch.allclose(logits[:1], single, atol=1e-4),
        f"max |diff| {(logits[:1] - single).abs().max():.1e}",
    )
    labels = [pipe.segment(load_thermogram(p)) for p in paths]
    left_first = []
    for lab in labels:
        xs1, xs2 = np.where(lab == 1)[1], np.where(lab == 2)[1]
        if len(xs1) and len(xs2):
            left_first.append(xs1.mean() < xs2.mean())
    consistent = max(np.mean(left_first), 1 - np.mean(left_first))
    check(
        "C4 left/right labels are side-consistent",
        consistent > 0.9,
        f"label 1 on image-left in {np.mean(left_first):.0%} of {len(left_first)} frames; this "
        "checkpoint was trained with sides assigned before flips (fixed in ThermalDataset), so "
        "only the breast-vs-background mask is used downstream",
        warn=True,
    )
    dices = [dice(lab > 0, gt) for lab, gt in zip(labels, gt_masks)]
    check(
        "C5 Dice vs DMR-IR annotations",
        np.mean(dices) > 0.7,
        f"mean {np.mean(dices):.3f}, min {np.min(dices):.3f} (GT also covers the cleavage)",
        warn=True,
    )

    # ------------------------------------------------------------------ D. VICReg backbone
    bb = pipe.backbone
    check(
        "D1 backbone weights",
        not bb.training,
        f"sha256 {sha256(BACKBONE)}, strict load ok, eval mode",
    )
    x = torch.cat([embedding_input(load_thermogram(p)) for p in paths]).to(device)
    with torch.no_grad():
        conv = bb.conv1(bb.padding(x))  # VICReg ResNet pads before conv1
        mismatch = (
            ((conv.mean((0, 2, 3)) - bb.bn1.running_mean) / bb.bn1.running_var.sqrt())
            .abs()
            .mean()
            .item()
        )
        emb = bb(x)
        emb_single = bb(x[:1])
    check(
        "D2 input matches pretraining statistics",
        mismatch < 0.25,
        f"conv1 mean mismatch {mismatch:.3f} (ImageNet-normalised input: ~0.9)",
    )
    check(
        "D3 embeddings finite/non-negative",
        torch.isfinite(emb).all().item() and (emb >= 0).all().item(),
        f"{tuple(emb.shape)}",
    )
    check(
        "D4 eval-mode batch invariance",
        torch.allclose(emb[:1], emb_single, atol=1e-4),
        f"max |diff| {(emb[:1] - emb_single).abs().max():.1e}",
    )
    s = torch.linalg.svdvals(emb - emb.mean(0))
    p = s**2 / (s**2).sum()
    erank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    check(
        "D5 embedding effective rank",
        erank > 16,
        f"{erank:.1f} on {len(paths)} frames (low = dimensional collapse)",
        warn=True,
    )

    # ------------------------------------------------------------------ E. metrics
    g = torch.Generator().manual_seed(0)
    logits_m = torch.randn(6, 1, 32, 32, generator=g)
    target = (torch.rand(6, 1, 32, 32, generator=g) > 0.6).float()
    ours = get_metrics(logits_m, target)
    y, pr = target.flatten().numpy(), (logits_m.sigmoid() > 0.5).flatten().numpy()
    ref = {
        "iou": skm.jaccard_score(y, pr),
        "precision": skm.precision_score(y, pr),
        "recall (sensitivity)": skm.recall_score(y, pr),
        "f1": skm.f1_score(y, pr),
        "specificity": skm.recall_score(y, pr, pos_label=0),
        "auc": skm.roc_auc_score(y, logits_m.sigmoid().flatten().numpy()),
    }
    worst = max(abs(ours[k] - v) for k, v in ref.items())
    check(
        "E1 get_metrics == scikit-learn", worst < 1e-4, f"max |diff| {worst:.1e} over {list(ref)}"
    )
    cm = confusion_metrics(torch.from_numpy(pr.astype(np.float32)), torch.from_numpy(y))
    worst = max(
        abs(cm[k] - v)
        for k, v in [
            ("Precision", ref["precision"]),
            ("Recall", ref["recall (sensitivity)"]),
            ("F1", ref["f1"]),
            ("IoU", ref["iou"]),
            ("Specificity", ref["specificity"]),
        ]
    )
    check("E2 confusion_metrics == scikit-learn", worst < 1e-4, f"max |diff| {worst:.1e}")
    m = target.clone()
    check(
        "E3 Dice/boundary IoU extremes",
        abs(get_metrics(m * 20 - 10, m)["dice"] - 1) < 1e-6
        and abs(get_metrics(m * 20 - 10, m)["boundary_iou"] - 1) < 1e-6,
        "identical masks -> 1.0",
    )
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    probs = np.clip(y_true * 0.3 + rng.random(200) * 0.7, 0, 1)
    ours = binary_scores(y_true, probs)
    ref_auc = skm.roc_auc_score(y_true, probs)
    ref_f1 = skm.f1_score(y_true, probs > 0.5)
    check(
        "E4 classifier scores == scikit-learn",
        abs(ours["auc"] - ref_auc) < 1e-9 and abs(ours["f1"] - ref_f1) < 1e-9,
        f"ROC AUC from probabilities {ours['auc']:.3f}, F1 at 0.5 {ours['f1']:.3f}",
    )

    # ------------------------------------------------------------------ F. end-to-end
    out = pipe(paths[0])
    out_nomask = pipe.embed(load_thermogram(paths[0]))
    check(
        "F1 end-to-end run",
        out.embedding.shape == (2048,) and out.labels.shape == load_thermogram(paths[0]).shape,
        f"labels {out.labels.shape}, breast fraction {out.breast_mask.mean():.2f}, "
        f"embedding {out.embedding.shape}",
    )
    cos = float(
        np.dot(out.embedding, out_nomask)
        / (np.linalg.norm(out.embedding) * np.linalg.norm(out_nomask))
    )
    check(
        "F2 background masking changes the embedding",
        cos < 0.999,
        f"cosine(masked, unmasked) {cos:.3f}",
    )

    image = load_thermogram(paths[0])
    bench_input = encoder_input(image, out.breast_mask, "masked")
    diff = (bench_input - embedding_input(image, out.breast_mask)[0]).abs().max().item()
    check(
        "F3 benchmark input == pipeline inference input",
        diff == 0,
        f"masked encoder input, max |diff| {diff:.1e}",
    )

    counts = {s: sum(r[0] == s for r in results) for s in ("PASS", "WARN", "FAIL")}
    lines = [
        "# Pipeline check",
        "",
        f"Subset: {len(items)} DMR-IR frames from {2 * N_PATIENTS_PER_CLASS} patients. {counts}",
        "",
        "| Status | Check | Detail |",
        "|---|---|---|",
    ]
    lines += [f"| {s} | {n} | {d} |" for s, n, d in results]
    Path("runs").mkdir(exist_ok=True)
    Path("runs/pipeline_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{counts}")


if __name__ == "__main__":
    main()
