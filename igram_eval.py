#!/usr/bin/env python3
"""
igram_eval.py

Walks the igrams directory tree, finds every JPEG with a matching ground-truth
.oln file (valid JSON, DFTFringe format), runs the igram_finder detection
pipeline, and reports accuracy vs ground truth.

Status codes per image:
  OK       — detection succeeded; Δcenter and Δradius shown
  FAIL     — detection pipeline returned None
  SKIP     — no matching .oln, or .oln not valid JSON / wrong schema
  OVEREXP  — image is overexposed (saturation fraction ≥ threshold);
             detection is still attempted and results shown

Usage:
    python igram_eval.py [root_dir] [--overexp-thresh T] [--scale N]
                         [--threshold N] [--verbose] [--summary-only]

Defaults:
    root_dir         igrams/
    --overexp-thresh 0.10   (10 % of pixels ≥ 250)
    --scale          4
    --threshold      20
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Reuse detection and I/O from igram_finder
sys.path.insert(0, str(Path(__file__).parent))
from igram_finder import detect_mirror, load_oln_circle


# ---------------------------------------------------------------------------
# Overexposure detection
# ---------------------------------------------------------------------------

SATURATION_PIXEL = 250  # pixel value considered "saturated"


def overexposure_fraction(image_path: Path, scale: int = 4) -> float:
    """
    Return the fraction of pixels that are at or above SATURATION_PIXEL.
    Works on a downsampled (scale) grayscale image for speed.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    if scale > 1:
        h, w = img.shape
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    return float(np.mean(img >= SATURATION_PIXEL))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    path: Path
    status: str          # OK / FAIL / SKIP / OVEREXP
    skip_reason: str = ""
    detected: Optional[tuple] = None   # (cx, cy, r)
    ground_truth: Optional[tuple] = None  # (cx, cy, r)
    d_center: Optional[float] = None
    d_radius: Optional[float] = None
    sat_frac: Optional[float] = None   # saturation fraction (if checked)


@dataclass
class DirStats:
    directory: Path
    results: list[ImageResult] = field(default_factory=list)

    def ok_results(self):
        return [r for r in self.results if r.status in ("OK", "OVEREXP") and r.d_center is not None]

    def summary(self) -> dict:
        ok = self.ok_results()
        fails = sum(1 for r in self.results if r.status == "FAIL")
        skips = sum(1 for r in self.results if r.status == "SKIP")
        overexp = sum(1 for r in self.results if r.status == "OVEREXP")
        d_centers = [r.d_center for r in ok]
        d_radii   = [r.d_radius for r in ok]
        return {
            "compared": len(ok),
            "failed": fails,
            "skipped": skips,
            "overexposed": overexp,
            "d_center_mean": float(np.mean(d_centers)) if d_centers else None,
            "d_center_max":  float(np.max(d_centers))  if d_centers else None,
            "d_radius_mean": float(np.mean(d_radii))   if d_radii else None,
            "d_radius_max":  float(np.max(d_radii))    if d_radii else None,
        }


# ---------------------------------------------------------------------------
# Core walk + eval logic
# ---------------------------------------------------------------------------

def is_valid_oln(path: Path) -> bool:
    """Return True iff path is a valid DFTFringe .oln JSON file."""
    try:
        with open(path) as f:
            d = json.load(f)
        o = d["outside_outline"]
        _ = float(o["center_x"]), float(o["center_y"]), float(o["radius_x"])
        return True
    except Exception:
        return False


def describe_oln(path: Path) -> str:
    """Return a human-readable description of why an .oln file was skipped."""
    try:
        raw = path.read_bytes()
        # Old DFTFringe binary format: starts with 4 little-endian doubles (cx, cy, rx, ry)
        if len(raw) >= 32 and raw[0] not in (ord('{'), ord(' '), ord('\t'), ord('\n')):
            import struct
            cx, cy, rx, ry = struct.unpack_from('<4d', raw, 0)
            return f"binary .oln (cx={cx:.1f} cy={cy:.1f} r={rx:.1f})"
        return "not valid JSON / wrong schema"
    except Exception:
        return "unreadable"


def eval_directory(
    directory: Path,
    scale: int,
    threshold: int,
    overexp_thresh: float,
    verbose: bool,
) -> DirStats:
    stats = DirStats(directory=directory)

    # Collect JPEGs, sorted for consistent prev_circle chaining
    images = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg") and p.is_file()
    )

    if not images:
        return stats

    prev_circle: Optional[tuple] = None

    for img_path in images:
        oln_path = img_path.with_suffix(".oln")

        # --- Check for ground-truth .oln ---
        if not oln_path.exists():
            stats.results.append(ImageResult(
                path=img_path, status="SKIP",
                skip_reason="no .oln file",
            ))
            if verbose:
                print(f"    {img_path.name}  SKIP  (no .oln)")
            continue

        if not is_valid_oln(oln_path):
            reason = describe_oln(oln_path)
            stats.results.append(ImageResult(
                path=img_path, status="SKIP",
                skip_reason=reason,
            ))
            if verbose:
                print(f"    {img_path.name}  SKIP  ({reason})")
            continue

        gt = load_oln_circle(oln_path)

        # --- Overexposure check ---
        sat = overexposure_fraction(img_path, scale=scale)
        is_overexp = sat >= overexp_thresh

        # --- Detection ---
        detected = detect_mirror(
            img_path,
            scale=scale,
            threshold=threshold,
            prev_circle=prev_circle,
        )

        if detected is None:
            status = "FAIL"
            res = ImageResult(
                path=img_path, status=status,
                ground_truth=gt,
                sat_frac=sat if is_overexp else None,
            )
            stats.results.append(res)
            if verbose:
                overexp_tag = f"  [sat={sat:.1%}]" if is_overexp else ""
                print(f"    {img_path.name}  FAIL{overexp_tag}")
            continue

        cx, cy, r = detected
        d_center = float(np.hypot(cx - gt[0], cy - gt[1])) if gt else None
        d_radius = float(abs(r - gt[2])) if gt else None
        status = "OVEREXP" if is_overexp else "OK"

        res = ImageResult(
            path=img_path,
            status=status,
            detected=detected,
            ground_truth=gt,
            d_center=d_center,
            d_radius=d_radius,
            sat_frac=sat if is_overexp else None,
        )
        stats.results.append(res)
        prev_circle = detected

        if verbose:
            overexp_tag = f"  [sat={sat:.1%}]" if is_overexp else ""
            gt_str = (f"  gt cx={gt[0]:.1f} cy={gt[1]:.1f} r={gt[2]:.1f}" if gt else "")
            d_str = (
                f"  Δcenter={d_center:.1f}px  Δr={d_radius:.1f}px"
                if d_center is not None else ""
            )
            print(
                f"    {img_path.name}  {status:<7}"
                f"  cx={cx:.1f} cy={cy:.1f} r={r:.1f}"
                f"{gt_str}{d_str}{overexp_tag}"
            )

    return stats


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def walk_and_eval(
    root: Path,
    scale: int,
    threshold: int,
    overexp_thresh: float,
    verbose: bool,
    summary_only: bool,
) -> list[DirStats]:
    """Walk root (following symlinks), evaluate every leaf directory."""
    all_stats: list[DirStats] = []

    # Collect all directories that contain at least one JPEG with a valid .oln
    # Walk with followlinks=True
    dirs_to_eval: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()  # deterministic order
        p = Path(dirpath)
        jpegs = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg"))]
        olns  = [f for f in filenames if f.lower().endswith(".oln")]
        if jpegs and olns:
            dirs_to_eval.append(p)

    if not dirs_to_eval:
        print(f"No directories found with both JPEGs and .oln files under {root}")
        return all_stats

    print(f"Found {len(dirs_to_eval)} director(ies) with JPEG + .oln pairs\n")

    for d in dirs_to_eval:
        label = rel(d, root)
        if not summary_only:
            print(f"=== {label} ===")

        ds = eval_directory(d, scale, threshold, overexp_thresh, verbose and not summary_only)
        all_stats.append(ds)

        if not summary_only:
            s = ds.summary()
            parts = []
            if s["compared"]:
                parts.append(
                    f"compared={s['compared']}"
                    f"  Δcenter mean={s['d_center_mean']:.1f}px max={s['d_center_max']:.1f}px"
                    f"  Δr mean={s['d_radius_mean']:.1f}px max={s['d_radius_max']:.1f}px"
                )
            if s["failed"]:
                parts.append(f"FAIL={s['failed']}")
            if s["skipped"]:
                parts.append(f"skip={s['skipped']}")
            if s["overexposed"]:
                parts.append(f"overexp={s['overexposed']}")
            if not parts:
                parts.append("(nothing compared)")
            print(f"  → {', '.join(parts)}\n")

    return all_stats


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_stats: list[DirStats], root: Path) -> None:
    print("\n" + "=" * 78)
    print("OVERALL SUMMARY")
    print("=" * 78)

    # Group by top-level dataset (first component after root)
    from collections import defaultdict
    datasets: dict[str, list[DirStats]] = defaultdict(list)
    for ds in all_stats:
        try:
            rel_parts = ds.directory.relative_to(root).parts
        except ValueError:
            rel_parts = ds.directory.parts
        top = rel_parts[0] if rel_parts else str(ds.directory)
        datasets[top].append(ds)

    total_compared = total_failed = total_skipped = total_overexp = 0
    all_d_centers: list[float] = []
    all_d_radii:   list[float] = []

    for dataset_name in sorted(datasets):
        group = datasets[dataset_name]
        ok_results = [r for ds in group for r in ds.ok_results()]
        fails  = sum(ds.summary()["failed"]    for ds in group)
        skips  = sum(ds.summary()["skipped"]   for ds in group)
        overexp= sum(ds.summary()["overexposed"] for ds in group)

        d_centers = [r.d_center for r in ok_results]
        d_radii   = [r.d_radius for r in ok_results]

        total_compared += len(ok_results)
        total_failed   += fails
        total_skipped  += skips
        total_overexp  += overexp
        all_d_centers  += d_centers
        all_d_radii    += d_radii

        if d_centers:
            dc_str = (
                f"Δcenter mean={np.mean(d_centers):.1f}px  "
                f"p95={np.percentile(d_centers, 95):.1f}px  "
                f"max={np.max(d_centers):.1f}px"
            )
            dr_str = (
                f"Δr mean={np.mean(d_radii):.1f}px  "
                f"p95={np.percentile(d_radii, 95):.1f}px  "
                f"max={np.max(d_radii):.1f}px"
            )
        else:
            dc_str = dr_str = "—"

        fail_str = f"  FAIL={fails}" if fails else ""
        skip_str = f"  skip={skips}" if skips else ""
        overexp_str = f"  overexp={overexp}" if overexp else ""

        print(f"\n  {dataset_name}")
        print(f"    compared={len(ok_results)}{fail_str}{skip_str}{overexp_str}")
        if d_centers:
            print(f"    {dc_str}")
            print(f"    {dr_str}")

    print(f"\n  {'─'*60}")
    print(f"  TOTAL: compared={total_compared}  FAIL={total_failed}"
          f"  skip={total_skipped}  overexp={total_overexp}")
    if all_d_centers:
        print(
            f"  Δcenter mean={np.mean(all_d_centers):.1f}px  "
            f"p95={np.percentile(all_d_centers, 95):.1f}px  "
            f"max={np.max(all_d_centers):.1f}px"
        )
        print(
            f"  Δr      mean={np.mean(all_d_radii):.1f}px  "
            f"p95={np.percentile(all_d_radii, 95):.1f}px  "
            f"max={np.max(all_d_radii):.1f}px"
        )

    # Worst performers
    all_results = [r for ds in all_stats for r in ds.ok_results()]
    if all_results:
        worst_center = sorted(all_results, key=lambda r: r.d_center, reverse=True)[:5]
        worst_radius = sorted(all_results, key=lambda r: r.d_radius, reverse=True)[:5]
        print("\n  Worst Δcenter:")
        for r in worst_center:
            print(f"    {r.path.name}  Δcenter={r.d_center:.1f}px  Δr={r.d_radius:.1f}px"
                  f"  ({r.path.parent.name})")
        print("\n  Worst Δradius:")
        for r in worst_radius:
            print(f"    {r.path.name}  Δr={r.d_radius:.1f}px  Δcenter={r.d_center:.1f}px"
                  f"  ({r.path.parent.name})")

    # All FAIL images
    fail_results = [r for ds in all_stats for r in ds.results if r.status == "FAIL"]
    if fail_results:
        print("\n  FAILED images:")
        for r in fail_results:
            sat_str = f"  [sat={r.sat_frac:.1%}]" if r.sat_frac is not None else ""
            print(f"    {r.path.parent.name}/{r.path.name}{sat_str}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate igram_finder detection accuracy across all datasets "
            "by comparing against hand-placed .oln ground truth files."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="igrams",
        help="Root directory to search (default: igrams/)",
    )
    parser.add_argument(
        "--overexp-thresh",
        type=float,
        default=0.10,
        metavar="T",
        help="Fraction of pixels ≥ 250 to flag as overexposed (default: 0.10)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        metavar="N",
        help="Downsample factor for detection (default: 4)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        metavar="N",
        help="Brightness threshold for outer-envelope (default: 20)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-image detection details",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip per-directory listing; print only the summary table",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    all_stats = walk_and_eval(
        root=root,
        scale=args.scale,
        threshold=args.threshold,
        overexp_thresh=args.overexp_thresh,
        verbose=args.verbose,
        summary_only=args.summary_only,
    )

    print_summary(all_stats, root)


if __name__ == "__main__":
    main()