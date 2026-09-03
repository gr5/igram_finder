# igram_finder Algorithm

This document explains how `igram_finder.py` locates the circular mirror aperture
in a Bath interferogram JPEG, finds bright internal-reflection spots inside that
aperture, and encodes both as a DFTFringe `.oln` outline file.  It is written for
someone who already understands Bath interferometry and DFTFringe but wants to
know exactly what the code is doing and why each decision was made.

---

## Background: what a Bath interferogram looks like

A Canon DSLR shooting a red-laser Bath interferometer produces a 6000 × 4000 (or
3456 × 5184) JPEG in which the scene contains roughly three concentric zones:

```
+-------------------------------------------+
|   dark background                         |
|   +-----------------------------------+   |
|   | outer illumination envelope       |   |
|   |   (reference beam spill-over)     |   |
|   |   +---------------------------+   |   |
|   |   |  mirror aperture          |   |   |
|   |   |  (interference fringes)   |   |   |
|   |   +---------------------------+   |   |
|   +-----------------------------------+   |
+-------------------------------------------+
```

**Mirror aperture** — the circular region where the object beam (reflected from the
mirror) and the reference beam interfere.  This is the circle DFTFringe needs.
It is filled with bright-dark fringe bands whose spatial frequency depends on the
tilt between the two beams.

**Outer illumination envelope** — a larger illuminated disc produced by the
reference beam path.  It contains the mirror aperture but is NOT the mirror.
The envelope can itself show fringe-like structure in dense-fringe or low-tilt
sessions.

**Dark background** — everything outside the envelope; pixel values close to zero.

The mirror is typically 50–70 % of the envelope diameter.  The exact ratio depends
on the optical layout and changes slightly between sessions.

---

## Processing overview

All computation runs on a downsampled (scale = 4 by default, so 1500 × 1000 or
864 × 1296) grayscale copy of the image.  The final circle and any spot polygons
are scaled back to full-resolution pixels.

```
JPEG  →  grayscale  →  ¼-scale
         │
         ├── Stage 1: brightness threshold → outer envelope mask
         │
         ├── Stage 2: local variance map → fringe-region binary mask
         │              (+ outer-ring fallback if Stage 3 grabs the envelope)
         │              + fringe_score (5×5 variance / mean brightness)
         │
         ├── Stage 3: RANSAC circle fit on fringe-region contour
         │
         ├── Stage 4: radial gradient refinement
         │              (+ prev-circle fallback if radius jumps > 10 %)
         │
         ├── Stage 5: bright-spot (internal-reflection) detection
         │              (on by default; --no-spots disables)
         │              + prev-frame hysteresis seeding
         │
         ├── result (cx, cy, radius, fringe_score, regions) → .oln
         │
         └── optional second pass (--re-look):
                session consensus → re-anchor outliers (20–80 px window)
```

---

## Stage 1 — Outer illumination envelope

**Goal:** find a binary mask that covers the illuminated area and excludes
everything outside it.  This bounds the search for subsequent stages and prevents
bright artefacts outside the envelope from contaminating the fringe contour.

**Steps:**

1. `cv2.threshold(small, threshold=20, ...)` — binary threshold.  Pixels brighter
   than 20 (out of 255) are `1`; background is `0`.  The default of 20 is
   conservative; it works even for dim images where the fringes barely reach
   pixel value 30.

2. Morphological close with a 7 × 7 ellipse kernel — bridges the dark fringe
   bands that would otherwise fragment the envelope into many small pieces.

3. Morphological open with a 3 × 3 ellipse kernel — removes single-pixel noise
   introduced by the close.

4. `cv2.findContours` → take the largest contour by area → fill it to produce
   `envelope_mask`.

5. `cv2.minEnclosingCircle(largest_env)` → extract `r_env_s`, the envelope
   radius at small scale.  This value is used later by the outer-ring fallback.

**Why the largest contour?**  The mirror and envelope appear as one large bright
region; any other contours are small artefacts (lens flare, stray reflections).
Taking the maximum by area is robust.

---

## Stage 2 — Local variance mask

**Goal:** produce a binary mask covering the mirror's fringe region, so that the
Stage 3 RANSAC fits a circle to the mirror boundary and not the envelope boundary.

**Why variance?**  Interference fringes are a high-spatial-frequency brightness
modulation.  A local variance map measures how much the pixel values vary within a
neighbourhood: high variance → fringes present → inside the mirror.  Outside the
mirror but inside the envelope the illumination is smooth (single-beam), so
variance is low.

**Steps:**

1. Compute the local variance map using a 21 × 21 uniform filter:

   ```
   mean  = uniform_filter(I,    21)
   var   = uniform_filter(I²,   21) − mean²
   ```

   (The `uniform_filter` trick is equivalent to a sliding-window mean and is much
   faster than explicit looping.)

2. Zero out all pixels outside `envelope_mask`, so any high-variance artefacts
   beyond the envelope are ignored.

3. Otsu threshold on `sqrt(var)` — taking the square root linearises the
   distribution and gives better bimodal separation between the fringe region
   (high var) and the single-beam region (low var).  Otsu's algorithm then finds
   the threshold that maximises between-class variance.

4. Morphological close with a 15 × 15 ellipse — the dark bands of the fringe
   pattern appear as "holes" in the variance map (a dark fringe has near-zero
   local variance).  The close fills these gaps, turning the fringe region into a
   solid filled disc.

5. Morphological open with a 5 × 5 ellipse — removes small islands left by the
   close.

6. `cv2.findContours` → take the largest contour.  This is the boundary of the
   fringe/mirror region.

**Implementation note:** Stages 2 and 3 are wrapped together in the internal
helper `_fringe_ransac(mask)`, which accepts an arbitrary binary mask to zero the
variance map.  This makes it easy to call both stages a second time with a
different mask (see the outer-ring fallback).

---

## Stage 3 — RANSAC circle fit

**Goal:** fit a circle to the fringe-region contour in a way that is robust to
partial, uneven, or asymmetric coverage of the mirror's arc.

**Why RANSAC and not least-squares directly?**

A plain algebraic least-squares fit (the Kasa method used by `fit_circle_lsq`) is
fast but biased when the contour does not cover the full 360°.  In a typical
interferogram, parts of the mirror may be significantly dimmer than others —
especially near tilt nulls — so some arc segments are absent from the variance
contour.  RANSAC tolerates this: it repeatedly samples 3 random points, fits a
circle, and counts how many of the remaining points lie within `residual_threshold`
of that circle.  The hypothesis with the most inliers wins.

**Steps:**

1. Thin the contour to ≤ 400 points (stride = `len(pts) // 400`) to keep the
   1000-trial RANSAC fast while preserving good angular coverage.

2. For each of 1000 trials: pick 3 random points; fit a circle with `fit_circle_lsq`;
   reject if radius < 5 or > 1000 (small scale); count inliers within 5 px.

3. Refit with all inliers using `fit_circle_lsq` for maximum accuracy.

**Result:** a centre `(cx_s, cy_s)` and radius `r_s` at small scale.  At this
point `r_s` is typically 5–15 small-scale pixels (20–60 full-res pixels) larger
than the true mirror radius, because the morphological closing inflated the fringe
contour outward.  Stage 4 corrects this.

---

## Outer-ring fallback (between Stages 3 and 4)

**Problem:** In some sessions — particularly "ZeroDeg" (very low tilt, few fringes)
or dense-fringe sessions — the outer illumination envelope itself develops a
high-variance ring at its boundary.  Morphological closing then bridges this ring
to the mirror fringe region, producing a single merged contour that encompasses
the entire illuminated area.  Stage 3 RANSAC fits this merged contour and returns
`r_s ≈ r_env_s`, which is the envelope radius rather than the mirror radius.

**Detection:** `r_s > 0.90 × r_env_s`

**Fix:** Erode `envelope_mask` by `max(3, 5% × r_env_s)` pixels to produce
`inner_mask`, which excludes the outermost ring.  Call `_fringe_ransac(inner_mask)`
again.  The eroded mask excludes the high-variance outer ring while still covering
the full mirror aperture (the mirror is always well inside 90% of the envelope
radius across all tested datasets).  Accept the retry result only if its radius
is < 90% of `r_env_s` (i.e., it actually found something smaller than the
envelope).

**Why not always use the inner mask?**  The erosion shifts the Stage 3 centre
estimate slightly.  For images that work correctly first time, this shift can
push Stage 4's radial search into a different convergence basin, producing a
slightly worse result.  By applying the erosion only when Stage 3 clearly grabbed
the envelope, the fix is targeted and does not affect other datasets.

---

## Stage 4 — Radial gradient refinement

**Goal:** snap the circle to the true mirror edge with sub-pixel accuracy, and
correct the radius over-estimate left by Stage 3.

**Physical basis:** At the mirror boundary, the image transitions from a bright
fringe pattern (inside) to a smoother, dimmer region (outside).  After Gaussian
blurring to suppress the high-frequency fringe modulation, this transition appears
as a strong negative radial gradient — brightness drops sharply as radius
increases past the mirror edge.

**Steps:**

1. Gaussian blur the small-scale image with σ = 4 px.  This suppresses the fringe
   texture while preserving the mirror-edge brightness step.  (σ = 4 at ¼-scale
   is equivalent to σ = 16 full-resolution pixels, roughly one fringe period for
   typical setups.)

2. For each of 720 uniformly-spaced angles around the circle:

   a. Sample the blurred brightness along a radial line from `r_s − 20` to
      `r_s + 20` (small-scale pixels), centred on the Stage 3 estimate.

   b. Compute the discrete gradient of this 1D brightness profile with
      `np.gradient`.

   c. Find `best_i = argmin(gradient)` — the point of steepest brightness
      decrease going outward.

   d. Accept this as an edge point only if `gradient[best_i] < −0.2` (a minimum
      contrast threshold that rejects directions where the mirror edge is not
      visible, e.g. where the mirror overlaps a bright artefact).

3. Collect all accepted edge points as `(x, y)` coordinates.

4. Run a second RANSAC on the edge points (residual threshold = 3 px, tighter
   than Stage 3 because the data is already close to the answer) to get the
   refined circle.

**Why does this produce the right radius?**  The blurred image has a sigmoid-shaped
transition at the mirror edge.  The gradient peaks at the inflection point of this
sigmoid — the midpoint of the transition zone.  DFTFringe's outline convention is
to place the circle at the transition midpoint ("splitting the edge pixels"), so
the gradient peak naturally lands at the correct position.  Empirically this gives
mean radius error < 2 full-resolution pixels on all tested datasets.

**Search band:** The ±20 small-scale pixel search (±80 full-resolution pixels)
is wide enough to accommodate the Stage 3 over-estimate (~10–20 small-scale px)
with margin, but narrow enough to avoid the next fringe ring or the envelope
boundary.

---

## Stage 5 — Bright internal-reflection spot detection

**Goal:** find compact bright glows *inside* the mirror aperture that come from a
stray internal reflection in the optical system, and emit them as polygon
`regions` in the `.oln` so DFTFringe ignores those pixels.

Spot detection is **on by default**; `--no-spots` disables it.  It runs after
Stage 4 (`detect_spots()` is called from the end of `detect_mirror()`) so that it
can use the refined circle to bound the search.

**Physical basis:** an internal reflection lays a smooth bright glow on top of the
fringes.  Measured against hand-drawn ground-truth regions it is ~50–65
full-resolution px across (~7 % of the mirror radius), typically 20–28 % of the
radius out from centre.  Unlike the fringes it is *low* spatial frequency, and
unlike the mirror's overall illumination gradient it is *compact* — that pair of
properties is what separates it from everything else in the aperture.

**Steps** (all at detection scale unless noted):

1. Gaussian blur with σ = 6 → `lp`.  This suppresses the fringe modulation while
   leaving the broader glow intact.  (Stage 4 uses σ = 4 for the edge step; the
   spot search blurs harder because it must erase fringes completely, not just
   soften them.)

2. Smooth background estimate `bg`: morphological opening of `lp` with an ellipse
   kernel of ≈ 0.35 × `r_s`.  A kernel larger than any plausible spot erases the
   spots but still follows the mirror's slow brightness field.  The opening is
   computed on a 2×-downsampled copy (both image and kernel shrink) and resized
   back — ~16× faster than opening at detection scale, with no meaningful
   accuracy loss, since the background is smooth by construction.

3. Top-hat: `tophat = lp − bg`.  Flag a pixel when

   ```
   tophat > threshold × bg          (threshold = SPOT_THRESHOLD = 0.35)
   ```

   The comparison is against the *local* background rather than an absolute
   level, which makes it independent of exposure and of the mirror's overall
   brightness gradient.

4. Restrict the search to inside the mirror minus a thin rim (`rim_frac = 0.07`
   of the radius).  The brightness step at the mirror edge would otherwise
   produce a false top-hat ring all the way around.

5. **Hysteresis from the previous frame:** the previous frame's polygons are
   rasterised, dilated with a 31 × 31 ellipse, and inside that neighbourhood the
   threshold is relaxed to `_SPOT_PREV_RELAX` (0.6) × threshold.  A spot that
   dims slightly between frames is therefore still caught.

6. Morphological close (5 × 5), then dilate (5 × 5), so the polygon generously
   covers the spot's halo.  Erring large is safe — DFTFringe simply ignores the
   region.

7. Connected components.  Keep a blob only if
   `_SPOT_MIN_AREA` (30 px) ≤ area ≤ π · (`_SPOT_MAX_RADIUS_FRAC` · `r_s`)²
   with `_SPOT_MAX_RADIUS_FRAC = 0.12`.  Both bounds are expressed relative to
   the (also scaled) mirror radius, so they are scale-invariant.

8. Contour each surviving blob → `cv2.approxPolyDP` at 1 % of the arc length →
   polygon, with vertices multiplied by `scale` back to full resolution.

9. **Swarm rejection:** if more than `_SPOT_MAX_COUNT` (3) polygons survive,
   return *none* of them.  A genuine reflection produces one or two spots; a
   frame yielding many is a low-tilt fringe storm.

**Why 0.35?**  The hand-curated ground truth separates cleanly in this measure:

| Spot class | Excess over local background |
|---|---|
| Bright / damaging (every curated primary) | 45 – 76 % |
| Subtle secondary | 13 – 22 % |

A default of 0.35 catches every damaging spot and skips the subtle ones;
`--spot-threshold 0.15` pulls the subtle ones in as well.  The value is a CLI knob
precisely because the line between "harmless" and "damaging" is not yet settled.

**Why the area cap and swarm rejection?**  A naive top-hat fired on 38 of 323
images in the clean (no-defect) datasets, almost all of them low-tilt or
zero-tilt frames whose widely-spaced fringes survive the σ = 6 blur and read as
bright blobs.  The two filters cut spurious detections from 221 spots to 37 while
retaining 17/17 curated primary spots, with detected centroids within 1–13 px of
the hand-drawn ones.  Most of the 37 residual detections are not in fact false:
they surfaced a genuine, previously-unlabelled reflection appearing at a stable
image location (33 % of the radius from centre) in every rotation subdirectory of
one session — the signature of an optically-fixed reflection.

**Cross-frame seeding:** because the reflection is fixed in the optical system,
its position is roughly stable across a session (it can drift a little and change
intensity with the setup).  `process_directory` therefore chains `prev_spots`
from frame to frame, exactly as it chains `prev_circle`; when an image is skipped
because it already has an `.oln`, `load_oln_regions()` seeds the chain from that
file instead.  The first image of a directory has no seed, so a spot sitting
right at threshold may be missed there and caught on later frames.

**Interaction with `--re-look`:** the second pass re-anchors the mirror circle
only.  Whatever regions the first pass found are carried through unchanged.

**Cost:** ~0.05 s/image, thanks to the downsampled background opening.

---

## Fringe visibility score

Computed alongside the variance map using a 5 × 5 uniform filter (faster
spatial frequency than the 21 × 21 detection filter):

```
sf5   = uniform_filter(I,   5)
var5  = uniform_filter(I²,  5) − sf5²
fringe_score = mean(sqrt(var5)  inside envelope)
             / mean(I           inside envelope)
```

The score is a dimensionless ratio.  Typical values:

| Image type | fringe_score |
|---|---|
| High-fringe (many tight bands) | 0.30 – 0.45 |
| Normal (6–12 fringes) | 0.18 – 0.32 |
| Low-tilt (1–3 fringes) | 0.11 – 0.14 |
| Near-zero tilt (0–1 fringes) | 0.10 – 0.13 |

Because low-tilt and normal-low-contrast images overlap around 0.11–0.13,
a fixed global threshold cannot reliably classify individual images.  The
score is therefore reported in the output but not used to gate detection —
detection is always attempted.  `LOW_FRINGE_SCORE = 0.10` is a conservative
floor that flags only the most extreme cases.

The score is used by the re-look second pass to identify which images are
reliable enough to contribute to the session consensus.

---

## Re-look second pass (optional, `--re-look`)

**Problem:** In a directory that spans multiple rotations (or occasionally a
badly-failed first-pass image), a minority of detections may land 20–80 px
from the true position.  If the true mirror position is stable for the
majority of frames, the session consensus can be used to re-anchor these
outliers.

**Algorithm:**

1. **Identify high-fringe images:** those with `fringe_score ≥ LOW_FRINGE_SCORE`
   (0.10).  These are reliable enough to contribute to consensus.

2. **Compute session consensus:** median `(cx, cy, radius)` across all
   high-fringe detections.  The median is robust to the 10–20 % of frames
   that are typically low-fringe or wrong.

3. **For each high-fringe image:**

   a. Compute `d = |detected_center − consensus_center|` (pixels).

   b. If `d < _RELOOK_MIN_PX` (20 px): already close; skip.

   c. If `d > _RELOOK_MAX_PX` (80 px): mirror has genuinely moved (e.g. new
      rotation group); skip — never force a cross-group fit.

   d. Otherwise: reload the image, run Stage 4 (`refine_by_radial_gradient`)
      anchored on the consensus circle.

   e. Accept the re-look result if `(d_new + Δr_new) < (d_old + Δr_old) × 0.9`
      (reduces combined distance + radius error by ≥ 10 %).  Overwrite the
      `.oln` file (unless `--dry-run`).

**Safety bounds:** The 80 px cap prevents forcing a consensus from one
rotation group onto a genuinely repositioned group.  In the tested datasets
(AT8IN 2024-08-18, two groups ~87 px apart) all group-2 images cleared the
80 px limit and were correctly left alone.

**Typical result:** Low-fringe outliers that drifted 20–40 px from the true
position are corrected.  Near-zero-tilt images are also in this range but
Stage 4 may not converge from the consensus seed if the mirror edge has no
brightness gradient; those re-looks are rejected by the 10 % criterion.

---

## Prev-circle fallback (after Stage 4)

**Problem:** In a small number of frames within an otherwise good sequence, the
variance mask can fail silently — Stage 3 produces a plausible-looking circle that
is actually the outer envelope, and Stage 4 refines it to the envelope boundary
rather than the mirror.  When processing a sequence of images of the same physical
mirror, the mirror radius should be nearly constant between frames.  A sudden jump
in `r_s` of more than 10% signals a failure.

**Detection:** `|r_s − prev_r_s| > 0.10 × prev_r_s`

**Fix:** Re-run Stage 4 (`refine_by_radial_gradient`) from the previous frame's
circle `(prev_cx_s, prev_cy_s, prev_r_s)`.  Because the mirror is the same
physical object, the gradient search anchored on the correct radius converges to
the true mirror edge even when the variance stage failed.

**Acceptance condition:** Accept the fallback result only if its radius is also
within 10% of `prev_r_s`.  If the fallback also diverges, emit a WARNING but keep
the original (possibly wrong) Stage 4 result rather than silently substituting a
worse one.

**Scope:** This fallback requires a `prev_circle` and therefore cannot help the
first image in a directory.  The outer-ring fallback (above) handles the
first-image case.

---

## Circle fitting: Kasa least-squares

`fit_circle_lsq` uses the algebraic (Kasa) circle fit, which rewrites the circle
equation

```
(x − cx)² + (y − cy)² = r²
```

as a linear system

```
2·cx·x + 2·cy·y + c = x² + y²
```

where `c = r² − cx² − cy²`.  Solving with `numpy.linalg.lstsq` gives `(cx, cy, c)`
from which `r = sqrt(c + cx² + cy²)`.  This is fast (O(n) per fit) and exact for
points on a true circle, making it ideal for the inner RANSAC loop.

---

## .oln file format

DFTFringe outline files are JSON with the following schema:

```json
{
    "dft_filter_radius": 25,
    "inside_outline":  { "center_x": -1, "center_y": -1, "radius_x": 0, "radius_y": 0 },
    "outside_outline": { "center_x": cx, "center_y": cy, "radius_x": r, "radius_y": r },
    "regions": [ [ {"x": x0, "y": y0}, {"x": x1, "y": y1}, ... ] ]
}
```

- `outside_outline` — the mirror aperture circle.  `radius_x == radius_y` because
  DFTFringe models the outline as an ellipse but we always write a circle.
- `inside_outline` — an inner obstruction mask (secondary mirror, etc.).  We write
  `cx = cy = −1` (disabled) because the mirrors tested are solid primaries.
- `dft_filter_radius` — controls DFTFringe's Fourier filter, unrelated to the
  aperture circle.  We always write 25.
- `regions` — a list of polygonal exclusion zones that DFTFringe ignores.  Stage 5
  writes one polygon per detected bright internal-reflection spot, each a list of
  `{"x", "y"}` vertices in full-resolution pixels.  The list is empty when no
  spots are found and when `--no-spots` is used.

All coordinates are in full-resolution pixels, with the origin at the top-left of
the image (standard image convention, which DFTFringe uses).

An older binary format also exists (pre-JSON DFTFringe): 4 little-endian IEEE 754
doubles at offset 0 = cx, cy, rx, ry, followed by more data.  `igram_eval.py`
detects this format and reports the decoded values in the skip message, but does
not use binary files as ground truth.

---

## Performance

Tested on 6000 × 4000 Canon DSLR JPEGs, Intel Core i5, no GPU:

- Image load + grayscale conversion: ~0.15 s
- Stages 1–3 (threshold, variance, RANSAC): ~0.10 s
- Stage 4 (720-angle gradient search + RANSAC): ~0.15 s
- Stage 5 (blur, downsampled opening, blob filtering): ~0.05 s
- **Total: ~0.45 s per image** (~0.4 s with `--no-spots`)

The bottleneck is Stage 4's Python loop over 720 angles.  Vectorising this with
NumPy would reduce it to ~0.05 s but the current speed is acceptable for batch
processing of a typical session (30–50 images ≈ 15–25 s total).

---

## Accuracy (validated 2026-03-27, 358 ground-truth images across 9 datasets)

| Dataset | n | Δcenter mean | Δcenter p95 | Δr mean | Δr p95 |
|---|---|---|---|---|---|
| 150mm-f4 (4 sessions) | 84 | 1.5 px | 3.5 px | 1.9 px | 2.9 px |
| 200mm-f6 | 66 | 2.3 px | 8.7 px | 1.1 px | 4.5 px |
| AT8IN | 89 | 2.1 px | 5.2 px | 2.7 px | 4.8 px |
| Dougs 10″ f/3.75 | 79 | 1.0 px | 2.0 px | 1.4 px | 3.5 px |
| Ken 6.5″ f/9 | 8 | 1.0 px | 1.8 px | 2.1 px | 3.2 px |
| Voltage=3.45 (200mm f/5.9) | 32 | 0.9 px | 1.8 px | 0.9 px | 1.6 px |
| **Overall** | **358** | **1.7 px** | **4.3 px** | **1.8 px** | **4.2 px** |

Ground truth is hand-placed DFTFringe outlines (JSON format).  Zero detection
failures across all 358 images.  All coordinates are full-resolution pixels.

---

## Known limitations and failure modes

**Low-fringe (near-zero-tilt) images:** When the interferometer is set to near-zero
tilt, fringes are few and widely spaced.  Local variance inside the mirror is
lower than in high-fringe images, and the outer illumination boundary sometimes
shows higher relative variance.  The outer-ring fallback handles this for the first
image; subsequent images benefit from the prev-circle fallback.  Accuracy degrades
slightly in these cases (worst-case ~12 px error vs ~2 px typical).

**Dense-fringe envelope contamination:** When the outer envelope itself fills with
high-variance fringe-like patterns (occurs in the 200mm-f6 "ZeroDeg" sessions due
to the reference-beam geometry), Stage 3 can latch onto the envelope.  The
outer-ring fallback corrects this automatically.

**No ground truth for first image:** Both fallbacks require a previous circle.  If
the very first image in a session has an unusual appearance (severe fringe
contrast, partial obstruction), the algorithm has no recovery mechanism.  In
practice this has not been observed across ~700 tested images.

**Spot false positives on low-tilt images:** The σ = 6 blur does not fully erase
the widely-spaced fringes of a near-zero-tilt frame, so their broad undulations
can read as bright blobs.  The area cap and swarm rejection remove the great
majority, but a single near-threshold spot can still be emitted on such a frame.
These are exactly the images DFTFringe does not process anyway; use `--no-spots`
if the spurious regions are unwanted.

**Subtle spots below the default threshold:** Secondary reflections sitting
13–22 % over the local background are ignored by the 0.35 default.  Whether they
need masking at all is not yet established; `--spot-threshold 0.15` includes them
at some risk of false positives.

**Over-exposure:** Saturated pixels (≥ 250 out of 255) clip the fringe contrast
and reduce variance.  `igram_eval.py` reports a saturation fraction for each image
but the main algorithm has no special handling — it will attempt detection but
accuracy may be reduced.