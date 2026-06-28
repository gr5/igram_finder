#!/usr/bin/env python3
"""
igram_finder.py

Automatically detects the circular mirror aperture in Bath interferogram
JPEG images and writes DFTFringe .oln files.

Detection pipeline:
  1. Outer envelope  — brightness threshold + largest contour.  Used to
     bound the search region and exclude bright artefacts outside the mirror.
  2. Variance mask   — local variance (fringes have high spatial-frequency
     content) + Otsu threshold + morphological closing.  Gives a binary mask
     roughly covering the mirror's fringe region.
  3. RANSAC circle fit — robust to the asymmetric fringe coverage that can
     bias a plain least-squares fit when some of the mirror's arc is dimmer.
     Gives a good center but radius typically ~10 px too large.
  4. Radial gradient refinement — for each of 720 angles, walks a short
     radial segment and finds the sharpest brightness drop (the mirror edge).
     A second RANSAC fit to those per-angle edge points converges to the
     circle that "splits the edge pixels" as DFTFringe expects.
  5. Bright-spot detection — finds compact bright glows inside the mirror that
     come from internal reflections in the optical system.  These are written
     as polygon "regions" in the .oln so DFTFringe ignores those pixels.

Typical accuracy on test data: Δcenter ≤ 5 px, Δradius ≤ 5 px in
full-resolution (6000 × 4000) images.  Processing time: ~0.4 s/image.

Usage:
    python igram_finder.py <directory> [options]

Options:
    --force        Overwrite existing .oln files (default: skip)
    --dry-run      Detect and report without writing any files
    --compare      Compare detection against existing .oln files (testing)
    --threshold N  Brightness threshold for outer-envelope detection (default: 20)
    --scale N      Downsample factor — 4 means work at 1/4 resolution (default: 4)
    --jump-warn N  Warn if circle center moves > N px between frames (default: 50)
    --re-look      Second pass: use session consensus to re-anchor outlier detections
    --no-spots     Disable bright internal-reflection spot detection (on by default)
    --spot-threshold F  Brightness excess over local background to flag a spot (default: 0.35)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Fringe visibility score below which an image is flagged as "low-fringe"
# (near-zero tilt).  These images are not suitable for DFTFringe processing.
LOW_FRINGE_SCORE: float = 0.10

# Re-look second pass: re-anchor an outlier only when its centre differs
# from session consensus by at least _RELOOK_MIN_PX (otherwise it's already
# close) and at most _RELOOK_MAX_PX (otherwise the mirror may have moved).
_RELOOK_MIN_PX: float = 20.0
_RELOOK_MAX_PX: float = 80.0

# Bright-spot (internal-reflection) detection.  A spot is flagged where the
# fringe-suppressed brightness rises above the smooth local background by at
# least SPOT_THRESHOLD (a fraction of that background).  Ground-truth bright
# spots sit 45-76 % over background; subtle harmless ones 13-22 %, so a default
# of 0.35 catches the damaging spots while ignoring the subtle ones.
SPOT_THRESHOLD: float = 0.35

# Near a spot found in the previous frame, the threshold is relaxed by this
# factor (hysteresis) so a spot that dims slightly between frames is still
# caught.  The same physical reflection persists across a session.
_SPOT_PREV_RELAX: float = 0.6

# Minimum blob area (in detection-scale pixels) to accept as a spot, rejecting
# isolated noise.  A real spot is ~12-16 px across at scale 4.
_SPOT_MIN_AREA: int = 30

# Maximum spot radius as a fraction of the mirror radius.  Internal reflections
# are compact (~7 % of the mirror radius in the ground truth); blobs larger than
# this are the broad brightness undulations of a low-tilt image, not a spot.
# Scale-invariant because it is relative to the (also-scaled) mirror radius.
_SPOT_MAX_RADIUS_FRAC: float = 0.12

# A genuine reflection produces one or two spots.  If a single frame yields more
# than this many candidates it is a low-tilt fringe-undulation storm, not a set
# of real reflections, so all of its candidates are rejected.
_SPOT_MAX_COUNT: int = 3


# ---------------------------------------------------------------------------
# Circle fitting helpers
# ---------------------------------------------------------------------------

def fit_circle_lsq(points: np.ndarray) -> tuple[float, float, float]:
    """
    Algebraic least-squares circle fit (Kasa method).
    Fast and accurate when edge coverage is good.
    Returns (cx, cy, radius).
    """
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = result
    radius = np.sqrt(max(0.0, c + cx ** 2 + cy ** 2))
    return float(cx), float(cy), float(radius)


def ransac_circle(
    points: np.ndarray,
    residual_threshold: float = 5.0,
    max_trials: int = 1000,
    rng_seed: int = 42,
) -> Optional[tuple[float, float, float]]:
    """
    RANSAC circle fit — robust to partial arc coverage and outliers.

    Repeatedly picks 3 random points, fits a circle, counts inliers
    (points within `residual_threshold` of the circle), keeps the best
    consensus set, then refits using all inliers.

    Returns (cx, cy, radius) or None if no good fit is found.
    """
    n = len(points)
    if n < 10:
        return None

    best_inlier_count = 0
    best_inliers: np.ndarray = np.empty(0, dtype=int)
    rng = np.random.default_rng(rng_seed)

    for _ in range(max_trials):
        idx = rng.choice(n, 3, replace=False)
        try:
            cx, cy, r = fit_circle_lsq(points[idx])
        except np.linalg.LinAlgError:
            continue
        if r < 5 or r > 1000:
            continue
        dists = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        inliers = np.where(dists < residual_threshold)[0]
        if len(inliers) > best_inlier_count:
            best_inlier_count = len(inliers)
            best_inliers = inliers

    if best_inlier_count < 5:
        return None

    # Refit with all inliers for maximum accuracy
    return fit_circle_lsq(points[best_inliers])




# ---------------------------------------------------------------------------
# Stage 4: radial gradient refinement
# ---------------------------------------------------------------------------

def refine_by_radial_gradient(
    small: np.ndarray,
    cx_s: float,
    cy_s: float,
    r_s: float,
    n_angles: int = 720,
    search_half: int = 20,
    blur_sigma: float = 4.0,
    grad_threshold: float = 0.2,
) -> tuple[float, float, float]:
    """
    Refine a circle estimate by sampling the brightness gradient along radial
    lines and snapping each sample to the sharpest inward drop — the mirror
    edge.  A second RANSAC fit to the per-angle edge points gives the refined
    circle.

    This is conceptually similar to DFTFringe's "auto-trace" but computed
    analytically rather than by exhaustive search.

    Parameters
    ----------
    small       : grayscale image at detection scale
    cx_s, cy_s  : initial circle centre (small-scale pixels)
    r_s         : initial radius (small-scale pixels)
    n_angles    : number of radial probes around the circle (default 720)
    search_half : half-width of the radial search band in pixels (default 20)
    blur_sigma  : Gaussian sigma to suppress fringe texture before gradient
                  computation.  sigma=4 suppresses fringes while preserving
                  the mirror-edge brightness step (default 4)
    grad_threshold : minimum negative gradient to accept an edge point;
                     rejects noisy/featureless directions (default 0.2)

    Returns (cx_s, cy_s, r_s) — possibly unchanged if refinement fails.
    """
    blurred = cv2.GaussianBlur(small.astype(np.float32), (0, 0), blur_sigma)
    sh, sw = small.shape
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    r_lo = max(5, int(r_s - search_half))
    r_hi = int(r_s + search_half)
    radii = np.arange(r_lo, r_hi + 1)

    edge_pts: list[tuple[float, float]] = []
    for ca, sa in zip(cos_a, sin_a):
        xs = np.clip((cx_s + radii * ca).astype(int), 0, sw - 1)
        ys = np.clip((cy_s + radii * sa).astype(int), 0, sh - 1)
        grad = np.gradient(blurred[ys, xs].astype(float))
        best_i = int(np.argmin(grad))
        if grad[best_i] < -grad_threshold:
            edge_r = radii[best_i]
            edge_pts.append((cx_s + edge_r * ca, cy_s + edge_r * sa))

    if len(edge_pts) < 20:
        return cx_s, cy_s, r_s  # not enough edges — keep original estimate

    pts = np.array(edge_pts)
    step = max(1, len(pts) // 400)
    model = ransac_circle(pts[::step], residual_threshold=3.0)
    if model is None:
        return cx_s, cy_s, r_s
    return model


# ---------------------------------------------------------------------------
# Bright-spot (internal-reflection) detection
# ---------------------------------------------------------------------------

def detect_spots(
    small: np.ndarray,
    cx_s: float,
    cy_s: float,
    r_s: float,
    scale: int = 4,
    threshold: float = SPOT_THRESHOLD,
    prev_spots: Optional[list] = None,
    blur_sigma: float = 6.0,
    rim_frac: float = 0.07,
) -> list:
    """
    Find bright internal-reflection spots inside the mirror aperture.

    An internal reflection adds a compact, smooth bright glow on top of the
    fringes.  It is detected as a "top-hat": the fringe-suppressed brightness
    (`small` blurred to remove fringes) minus a smooth background estimate (a
    large morphological opening that erases compact spots but preserves the
    mirror's slow brightness gradient).  A pixel is flagged when that excess
    exceeds `threshold` times the local background — a relative measure that is
    independent of exposure and of the mirror's overall brightness gradient.

    Parameters
    ----------
    small       : grayscale image at detection scale
    cx_s, cy_s  : mirror centre (small-scale pixels)
    r_s         : mirror radius (small-scale pixels)
    scale       : downsample factor, used to convert polygons back to full res
    threshold   : minimum brightness excess over local background, as a
                  fraction of that background (default SPOT_THRESHOLD)
    prev_spots  : spots found in the previous frame (full-resolution polygons);
                  near these locations the threshold is relaxed (hysteresis)
    blur_sigma  : Gaussian sigma to suppress fringe texture (default 6)
    rim_frac    : fraction of the radius excluded at the mirror edge, where the
                  brightness step would otherwise create false top-hat (0.07)

    Returns a list of polygons (each a list of ``{"x", "y"}`` dicts in
    full-resolution pixels), one per detected spot.  Empty if none found.
    """
    h, w = small.shape
    lp = cv2.GaussianBlur(small.astype(np.float32), (0, 0), blur_sigma)

    # Smooth background: morphological opening with a kernel larger than any
    # spot removes the spots but follows the mirror's broad brightness field.
    # The background is smooth, so the opening is computed on a 2x-downsampled
    # copy (kernel and image both shrink) and resized back — ~16x faster than
    # opening at full detection scale, with no meaningful accuracy loss.
    ds = 2
    lp_ds = cv2.resize(lp, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
    ksz = max(3, int(r_s / ds * 0.35)) | 1
    bg_ds = cv2.morphologyEx(
        lp_ds, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)),
    )
    bg = cv2.resize(bg_ds, (w, h), interpolation=cv2.INTER_LINEAR)
    tophat = lp - bg

    # Search only inside the mirror, minus a thin rim.
    yy, xx = np.ogrid[:h, :w]
    search_r = r_s * (1.0 - rim_frac)
    mirror = (xx - cx_s) ** 2 + (yy - cy_s) ** 2 <= search_r ** 2

    hot = (tophat > threshold * bg) & mirror

    # Hysteresis: relax the threshold near spots seen in the previous frame.
    if prev_spots:
        prev_mask = np.zeros((h, w), np.uint8)
        for poly in prev_spots:
            try:
                p = np.array(
                    [[pt["x"] / scale, pt["y"] / scale] for pt in poly],
                    dtype=np.int32,
                )
            except (KeyError, TypeError):
                continue
            if len(p) >= 3:
                cv2.fillPoly(prev_mask, [p], 1)
        if prev_mask.any():
            prev_mask = cv2.dilate(
                prev_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
            )
            relaxed = (tophat > _SPOT_PREV_RELAX * threshold * bg) & mirror
            hot = hot | (relaxed & (prev_mask > 0))

    hot_u8 = hot.astype(np.uint8)
    # Close small gaps, then dilate slightly so the polygon generously covers
    # the spot halo (DFTFringe ignores the region, so erring large is safe).
    hot_u8 = cv2.morphologyEx(
        hot_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    hot_u8 = cv2.dilate(
        hot_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    # Reject blobs larger than a plausible reflection (low-tilt undulations).
    max_area = np.pi * (_SPOT_MAX_RADIUS_FRAC * r_s) ** 2

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(hot_u8, connectivity=8)
    regions: list = []
    for i in range(1, n_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < _SPOT_MIN_AREA or area > max_area:
            continue
        comp = (labels == i).astype(np.uint8)
        conts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not conts:
            continue
        c = max(conts, key=cv2.contourArea)
        eps = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        poly = [
            {"x": int(round(pt[0][0] * scale)), "y": int(round(pt[0][1] * scale))}
            for pt in approx
        ]
        if len(poly) >= 3:
            regions.append(poly)

    # A frame with too many candidates is a low-tilt fringe storm, not a set of
    # real reflections — reject them all.
    if len(regions) > _SPOT_MAX_COUNT:
        return []

    return regions


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_mirror(
    image_path: Path,
    scale: int = 4,
    threshold: int = 20,
    prev_circle: Optional[tuple[float, float, float]] = None,
    jump_warn: float = 50.0,
    find_spots: bool = True,
    spot_threshold: float = SPOT_THRESHOLD,
    prev_spots: Optional[list] = None,
) -> Optional[tuple[float, float, float, float, list]]:
    """
    Detect the mirror aperture circle in an interferogram image.

    Bath interferograms have two distinct regions:
      - A large outer illumination envelope (easy to threshold).
      - The actual mirror aperture inside it, filled with interference
        fringes.  High local variance marks the fringe region.

    The outer envelope provides a spatial bound; RANSAC on the fringe-region
    contour finds the mirror circle even when fringe brightness is uneven.

    Returns (center_x, center_y, radius, fringe_score, regions) in full-image
    pixels, or None.  fringe_score is a normalised measure of fringe
    visibility; values below LOW_FRINGE_SCORE indicate a near-zero-tilt
    (low-fringe) image that is not suitable for DFTFringe processing.  regions
    is a list of bright internal-reflection spot polygons (empty when
    find_spots is False or none are found); see detect_spots.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"    ERROR: could not read {image_path.name}")
        return None

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    small_w, small_h = w // scale, h // scale
    small = cv2.resize(img_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Stage 1: outer illumination envelope
    # ------------------------------------------------------------------
    _, binary = cv2.threshold(small, threshold, 255, cv2.THRESH_BINARY)
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_c)
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_o)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(f"    ERROR: no contours found in {image_path.name}")
        return None

    largest_env = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_env) < 0.01 * small_w * small_h:
        print(f"    WARNING: illuminated area too small in {image_path.name}")
        return None

    # Filled envelope mask — used to exclude artefacts outside the mirror
    envelope_mask = np.zeros((small_h, small_w), np.uint8)
    cv2.drawContours(envelope_mask, [largest_env], -1, 255, cv2.FILLED)

    # Estimate the envelope radius for the outer-ring fallback below.
    _, r_env_s = cv2.minEnclosingCircle(largest_env)

    # ------------------------------------------------------------------
    # Stage 2: local variance mask (fringe region)
    # ------------------------------------------------------------------
    sf = small.astype(np.float32)
    mean_f = uniform_filter(sf, 21)
    var_map = np.clip(uniform_filter(sf ** 2, 21) - mean_f ** 2, 0, None)

    # Fringe visibility score: normalised high-spatial-frequency variance
    # inside the envelope.  A 5×5 filter captures fringe texture (higher
    # frequency than the 21×21 filter) without slow brightness gradients.
    # Low score → few fringes → near-zero tilt image.
    _sf5 = uniform_filter(sf, 5)
    _var5 = np.clip(uniform_filter(sf ** 2, 5) - _sf5 ** 2, 0, None)
    _env_pixels = sf[envelope_mask > 0]
    _env_mean = float(_env_pixels.mean()) + 1e-6
    fringe_score = float(np.sqrt(_var5[envelope_mask > 0]).mean()) / _env_mean

    def _fringe_ransac(mask: np.ndarray) -> Optional[tuple[float, float, float]]:
        """
        Run Stage 2 (variance mask) + Stage 3 (RANSAC) with the given mask.
        Returns (cx_s, cy_s, r_s) in small-scale pixels, or None.
        """
        var_in = var_map.copy()
        var_in[mask == 0] = 0.0

        var_sqrt = np.sqrt(var_in)
        v_max = var_sqrt.max()
        if v_max < 1e-6:
            return None
        v_u8 = np.clip(var_sqrt / v_max * 255, 0, 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(v_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vt = (otsu_val / 255.0 * v_max) ** 2

        vbin = (var_in > vt).astype(np.uint8) * 255
        vbin = cv2.morphologyEx(
            vbin, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        vbin = cv2.morphologyEx(
            vbin, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        conts, _ = cv2.findContours(vbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not conts:
            return None
        fc = max(conts, key=cv2.contourArea)
        pts_fc = fc.squeeze(axis=1).astype(float)
        if len(pts_fc) < 20:
            return None
        step = max(1, len(pts_fc) // 400)
        return ransac_circle(pts_fc[::step], residual_threshold=5.0, max_trials=1000)

    # ------------------------------------------------------------------
    # Stage 2+3: first pass with the full envelope mask
    # ------------------------------------------------------------------
    model = _fringe_ransac(envelope_mask)
    if model is None:
        print(f"    ERROR: RANSAC failed on {image_path.name}")
        return None

    cx_s, cy_s, r_s = model

    # ------------------------------------------------------------------
    # Outer-ring fallback: sometimes the variance mask latches onto a
    # high-variance ring at the outer envelope boundary instead of the
    # mirror aperture inside it (e.g. dense-fringe ZeroDeg sessions).
    # Detected when Stage 3 returns a circle nearly as large as the
    # envelope.  Fix: erode the mask by ~5 % of r_env to exclude the
    # outermost ring, then redo Stage 2+3.  This is only applied when
    # Stage 3 clearly grabbed the envelope (r_s > 90 % of r_env_s), so
    # it does not affect datasets where Stage 3 already finds the mirror.
    # ------------------------------------------------------------------
    if r_s > r_env_s * 0.90:
        erode_r = max(3, int(r_env_s * 0.05))
        erode_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_r + 1, 2 * erode_r + 1)
        )
        inner_mask = cv2.erode(envelope_mask, erode_k)
        retry = _fringe_ransac(inner_mask)
        if retry is not None and retry[2] < r_env_s * 0.90:
            print(
                f"\n    outer-ring latch detected (r={r_s*scale:.0f} px ≈ r_env={r_env_s*scale:.0f} px)"
                f" — retrying with inner mask...",
                end="", flush=True,
            )
            cx_s, cy_s, r_s = retry

    # ------------------------------------------------------------------
    # Stage 4: radial gradient refinement
    # Snaps the circle to the actual mirror edge by sampling the brightness
    # gradient at each angle.  Converts the ~10 px radius over-estimate from
    # the variance contour into sub-pixel accuracy.
    # ------------------------------------------------------------------
    cx_s, cy_s, r_s = refine_by_radial_gradient(small, cx_s, cy_s, r_s)

    # ------------------------------------------------------------------
    # Fallback: if we have a previous frame's circle and the radius has
    # changed by more than 10 % (same physical mirror = same radius),
    # the variance stage probably latched onto the outer envelope instead
    # of the mirror.  Re-run Stage 4 directly from prev_circle — since
    # the mirror is the same physical object, the radius and often the
    # position are similar enough for the gradient search to snap to it.
    # ------------------------------------------------------------------
    if prev_circle is not None:
        prev_r_s = prev_circle[2] / scale
        if abs(r_s - prev_r_s) > prev_r_s * 0.10:
            print(
                f"\n    radius mismatch (got {r_s*scale:.1f} px, expected ~{prev_r_s*scale:.1f} px)"
                f" — retrying from previous frame...",
                end="", flush=True,
            )
            prev_cx_s = prev_circle[0] / scale
            prev_cy_s = prev_circle[1] / scale
            fallback = refine_by_radial_gradient(small, prev_cx_s, prev_cy_s, prev_r_s)
            # Accept the fallback only if it converged to a plausible radius
            if abs(fallback[2] - prev_r_s) < prev_r_s * 0.10:
                cx_s, cy_s, r_s = fallback
            else:
                print(
                    f"\n    WARNING: fallback also returned implausible radius "
                    f"({fallback[2]*scale:.1f} px); detection may be unreliable"
                )

    cx = cx_s * scale
    cy = cy_s * scale
    radius = r_s * scale

    # Warn if the result jumps a lot from the previous frame
    if prev_circle is not None:
        px, py, pr = prev_circle
        dist = np.hypot(cx - px, cy - py)
        r_diff = abs(radius - pr)
        if dist > jump_warn or r_diff > jump_warn:
            print(
                f"    NOTE: large change from previous — "
                f"center moved {dist:.1f}px, radius Δ{r_diff:.1f}px"
            )

    # Bright internal-reflection spots inside the (now refined) mirror circle.
    regions: list = []
    if find_spots:
        regions = detect_spots(
            small, cx_s, cy_s, r_s,
            scale=scale, threshold=spot_threshold, prev_spots=prev_spots,
        )

    return cx, cy, radius, fringe_score, regions


# ---------------------------------------------------------------------------
# .oln I/O
# ---------------------------------------------------------------------------

def load_oln_circle(oln_path: Path) -> Optional[tuple[float, float, float]]:
    """Read (cx, cy, radius) from an existing .oln file."""
    try:
        with open(oln_path) as f:
            d = json.load(f)
        o = d["outside_outline"]
        return float(o["center_x"]), float(o["center_y"]), float(o["radius_x"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def load_oln_regions(oln_path: Path) -> list:
    """
    Read the `regions` polygon list from an existing .oln file.

    Returns a list of polygons (each a list of ``{"x", "y"}`` dicts), or an
    empty list if the file is missing, unparseable, or has no regions.  Used to
    seed the previous-frame spot search when an image is skipped.
    """
    try:
        with open(oln_path) as f:
            d = json.load(f)
        regions = d.get("regions", [])
        return regions if isinstance(regions, list) else []
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def write_oln(
    oln_path: Path,
    cx: float,
    cy: float,
    radius: float,
    regions: Optional[list] = None,
    dft_filter_radius: int = 25,
) -> None:
    """
    Write a DFTFringe .oln file.

    `regions` is an optional list of polygons marking bright internal-reflection
    spots that DFTFringe should ignore.  Each polygon is a list of
    ``{"x": int, "y": int}`` vertices in full-resolution pixels.  When omitted,
    an empty list is written (no regions).
    """
    data = {
        "dft_filter_radius": dft_filter_radius,
        "inside_outline": {
            "center_x": -1,
            "center_y": -1,
            "radius_x": 0,
            "radius_y": 0,
        },
        "outside_outline": {
            "center_x": cx,
            "center_y": cy,
            "radius_x": radius,
            "radius_y": radius,
        },
        "regions": regions if regions is not None else [],
    }
    with open(oln_path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Second-pass re-look
# ---------------------------------------------------------------------------

def _second_pass(
    results: list[tuple],
    scale: int,
    dry_run: bool,
) -> list[tuple]:
    """
    Use session consensus from high-fringe images to re-anchor outlier
    detections.

    For each high-fringe image whose centre differs from the session median
    by 20–80 px, re-run Stage 4 (radial gradient refinement) anchored on
    the consensus circle.  Accept the new result only if it reduces the
    combined distance from consensus by ≥ 10 %.  Images outside the 20–80 px
    window are either already correct (< 20) or the mirror has genuinely
    moved (> 80) and must not be overridden.

    `results` is a list of (img_path, cx, cy, radius, fringe_score, regions)
    tuples in full-resolution pixels.  Returns an updated list with the same
    structure; .oln files are rewritten for any accepted re-looks (unless
    dry_run is True).  The re-look only re-anchors the mirror circle; any spot
    regions already found in the first pass are preserved unchanged.
    """
    high_fringe = [
        (cx, cy, r)
        for _, cx, cy, r, fs, _ in results
        if fs >= LOW_FRINGE_SCORE
    ]
    if len(high_fringe) < 3:
        return results  # not enough data for a reliable consensus

    cx_c = float(np.median([c[0] for c in high_fringe]))
    cy_c = float(np.median([c[1] for c in high_fringe]))
    r_c  = float(np.median([c[2] for c in high_fringe]))

    updated = []
    for img_path, cx, cy, r, fs, regions in results:
        if fs < LOW_FRINGE_SCORE:
            updated.append((img_path, cx, cy, r, fs, regions))
            continue

        d_old = float(np.hypot(cx - cx_c, cy - cy_c))
        dr_old = abs(r - r_c)

        if d_old < _RELOOK_MIN_PX or d_old > _RELOOK_MAX_PX:
            # Too close (already good) or too far (mirror repositioned)
            updated.append((img_path, cx, cy, r, fs, regions))
            continue

        # Reload image and re-run Stage 4 from consensus seed
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            updated.append((img_path, cx, cy, r, fs, regions))
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        small = cv2.resize(img_gray, (w // scale, h // scale),
                           interpolation=cv2.INTER_AREA)

        new_cx_s, new_cy_s, new_r_s = refine_by_radial_gradient(
            small, cx_c / scale, cy_c / scale, r_c / scale
        )
        new_cx = new_cx_s * scale
        new_cy = new_cy_s * scale
        new_r  = new_r_s  * scale

        d_new  = float(np.hypot(new_cx - cx_c, new_cy - cy_c))
        dr_new = abs(new_r - r_c)

        if (d_new + dr_new) < (d_old + dr_old) * 0.9:
            tag = " [dry-run]" if dry_run else ""
            print(
                f"    [re-look] {img_path.name}:"
                f" cx={cx:.1f} cy={cy:.1f} r={r:.1f}"
                f" → cx={new_cx:.1f} cy={new_cy:.1f} r={new_r:.1f}{tag}"
            )
            if not dry_run:
                write_oln(img_path.with_suffix(".oln"), new_cx, new_cy, new_r, regions)
            updated.append((img_path, new_cx, new_cy, new_r, fs, regions))
        else:
            updated.append((img_path, cx, cy, r, fs, regions))

    return updated


# ---------------------------------------------------------------------------
# Directory processing
# ---------------------------------------------------------------------------

def process_directory(
    directory: Path,
    force: bool = False,
    dry_run: bool = False,
    compare: bool = False,
    scale: int = 4,
    threshold: int = 20,
    jump_warn: float = 50.0,
    re_look: bool = False,
    find_spots: bool = True,
    spot_threshold: float = SPOT_THRESHOLD,
) -> None:
    """
    Process all JPEG images in `directory`.

    --compare mode: only process images that already have a .oln, and print
    detected values alongside existing values for accuracy checking.
    """
    images = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg") and p.is_file()
    )

    if not images:
        print(f"No JPEG images found in {directory}")
        return

    if compare:
        images = [p for p in images if p.with_suffix(".oln").exists()]
        if not images:
            print(f"No images with matching .oln files found in {directory}")
            return
        print(f"Comparing detection against {len(images)} existing .oln file(s)")
    else:
        print(f"Found {len(images)} image(s) in {directory}")

    prev_circle: Optional[tuple[float, float, float]] = None
    prev_spots: list = []
    processed = skipped = failed = 0
    # (img_path, cx, cy, radius, fringe_score, regions)
    all_results: list[tuple] = []

    for img_path in images:
        oln_path = img_path.with_suffix(".oln")

        if oln_path.exists() and not force and not compare:
            existing = load_oln_circle(oln_path)
            if existing:
                prev_circle = existing
            prev_spots = load_oln_regions(oln_path)
            print(f"  {img_path.name}: skipped (already has .oln)")
            skipped += 1
            continue

        print(f"  {img_path.name}: detecting...", end="", flush=True)

        result = detect_mirror(
            img_path,
            scale=scale,
            threshold=threshold,
            prev_circle=prev_circle,
            jump_warn=jump_warn,
            find_spots=find_spots,
            spot_threshold=spot_threshold,
            prev_spots=prev_spots,
        )

        if result is None:
            print(" FAILED")
            failed += 1
            continue

        cx, cy, radius, fringe_score, regions = result
        fringe_tag = " [low-fringe]" if fringe_score < LOW_FRINGE_SCORE else ""
        spot_tag = f" [{len(regions)} spot(s)]" if regions else ""

        if compare:
            existing = load_oln_circle(oln_path)
            if existing:
                ex, ey, er = existing
                d_center = np.hypot(cx - ex, cy - ey)
                d_radius = abs(radius - er)
                existing_regions = load_oln_regions(oln_path)
                print(
                    f" cx={cx:.1f} cy={cy:.1f} r={radius:.1f}  fringe={fringe_score:.2f} | "
                    f"existing cx={ex:.1f} cy={ey:.1f} r={er:.1f} | "
                    f"Δcenter={d_center:.1f}px Δr={d_radius:.1f}px{fringe_tag}"
                    f" | spots {len(regions)} vs {len(existing_regions)} curated"
                )
            else:
                print(f" cx={cx:.1f} cy={cy:.1f} r={radius:.1f}  fringe={fringe_score:.2f}{fringe_tag}{spot_tag}")
            prev_circle = (cx, cy, radius)
            prev_spots = regions
            processed += 1
            continue

        tag = " [dry-run]" if dry_run else ""
        print(f" cx={cx:.1f} cy={cy:.1f} r={radius:.1f}  fringe={fringe_score:.2f}{fringe_tag}{spot_tag}{tag}")

        if not dry_run:
            write_oln(oln_path, cx, cy, radius, regions)

        prev_circle = (cx, cy, radius)
        prev_spots = regions
        all_results.append((img_path, cx, cy, radius, fringe_score, regions))
        processed += 1

    if re_look and all_results:
        print("\n--- Re-look (second pass) ---")
        _second_pass(all_results, scale=scale, dry_run=dry_run)

    if compare:
        print(f"\nCompared: {processed} image(s), {failed} failed")
    else:
        print(
            f"\nDone: {processed} written, {skipped} skipped, {failed} failed"
            + (" (dry-run — nothing written)" if dry_run else "")
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-detect mirror aperture in Bath interferogram images "
            "and write DFTFringe .oln files."
        )
    )
    parser.add_argument(
        "directory",
        help="Directory containing JPEG interferogram images",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .oln files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect without writing files",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare detection against existing .oln files (for testing/tuning)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        metavar="N",
        help="Brightness threshold for outer-envelope detection 0-255 (default: 20)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        metavar="N",
        help="Downsample factor for detection (default: 4 = quarter resolution)",
    )
    parser.add_argument(
        "--jump-warn",
        type=float,
        default=50.0,
        metavar="N",
        help="Warn if circle moves > N pixels between frames (default: 50)",
    )
    parser.add_argument(
        "--re-look",
        action="store_true",
        help=(
            "Second pass: use session consensus from high-fringe images to "
            "re-anchor detections that differ from consensus by 20–80 px"
        ),
    )
    parser.add_argument(
        "--no-spots",
        action="store_true",
        help=(
            "Disable detection of bright internal-reflection spots "
            "(spot detection is on by default)"
        ),
    )
    parser.add_argument(
        "--spot-threshold",
        type=float,
        default=SPOT_THRESHOLD,
        metavar="F",
        help=(
            "Brightness excess over local background, as a fraction, to flag a "
            f"bright spot (default: {SPOT_THRESHOLD}; lower = more sensitive)"
        ),
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"ERROR: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    process_directory(
        directory,
        force=args.force,
        dry_run=args.dry_run,
        compare=args.compare,
        scale=args.scale,
        threshold=args.threshold,
        jump_warn=args.jump_warn,
        re_look=args.re_look,
        find_spots=not args.no_spots,
        spot_threshold=args.spot_threshold,
    )


if __name__ == "__main__":
    main()
