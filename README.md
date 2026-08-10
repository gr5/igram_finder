# igram_finder

This Python script is intended to be used with the DFTFringe program. It will automatically find the
circular mirror aperture in an interferogram photograph. It writes an outline file (ending in
`.oln`) readable by DFTFringe, which will then make it much easier for the DFTFringe user when
processing many interferogram images. The script then removes the drudgery of finding and checking
each image in the process of getting a mirror's best surface error measurement.

# Bath Interferometry

The Bath interferometer is a relatively inexpensive interferometry tool for measuring the surface
error of an astronomical mirror. (There may be other uses of the Bath interferometer, but that is
the primary use.) Compared to a Laser Unequal Path Interferometer (LUPI), it is more laborious, but
the LUPI requires a much more expensive setup, including an extremely stable optical bench, a laser
with a long spatial coherence and some precision reference optics.

On the other hand, the Bath interferometer can use a cheap laser pointer laser and other inexpensive
components. The tradeoff is that one ends up needing to take many interferogram images, then
processing them with [DFTFringe](https://github.com/githubdoe/DFTFringe/releases). In that
processing step, I have found that selecting the mirror outline in the image for *every single
image* is one of the more tedious aspects. Therefore, a program that could do that for me would be
very useful. More Bath links: [GR Store Bath](http://gr5.org/bath/), [Setting up and using a Bath
Interferometer](https://www.youtube.com/watch?v=-C6UvLoBQEs)

# The Script

`igram_finder` automatically locates the circular mirror aperture in Bath interferogram JPEG images
and writes the result as a DFTFringe `.oln` outline file.

Typical accuracy on Canon DSLR images (6000 × 4000): center error < 5 px, radius error < 5 px,
processing time ~0.4 s/image.  See `ALGORITHM.md` for a full description of the detection pipeline.

The script was entirely "vibe-coded" using Claude Code and many instructions by me, Charles Wright.

---

## Scripts

| Script                 | Purpose |
|------------------------|---------|
| `igram_finder.py`      | Detect mirror aperture and write `.oln` files for a directory of JPEGs |
| `igram_finder_gui.py`  | Tkinter GUI wrapper — add directories, set options, run the detector, and preview results interactively |
| `igram_eval.py`        | Walk a directory tree, compare detections against hand-placed ground-truth `.oln` files, and report accuracy |

---

## Requirements

- Python 3.10 or later
- [OpenCV](https://pypi.org/project/opencv-python/) (`cv2`)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/) (for `scipy.ndimage.uniform_filter`)
- [Pillow](https://python-pillow.org/) (`PIL` / `ImageTk`) — GUI only

---

## Setup: creating a virtual environment

```bash
# 1. Create the venv (once) (Note - you can name your venv anything; here it's `.venv`)
python3 -m venv .venv

# 2. Activate it
#    Linux / macOS:
source .venv/bin/activate
#    Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install opencv-python numpy scipy pillow
```

To leave the venv when you're done:

```bash
deactivate
```

### Saving and restoring the exact versions

If you want to pin the exact versions currently in use:

```bash
pip freeze > requirements.txt
```

To recreate the environment elsewhere:

```bash
pip install -r requirements.txt
```

This is not recommended for normal users, as there can be version conflicts. Just install the
dependencies listed above and you should be good.

---

## GUI

`igram_finder_gui.py` provides a point-and-click interface to the detector.
Launch it with:

```bash
./run_gui.sh
```
Note: you must be in the directory where the igram finder scripts are.

Or manually:

```bash
source .venv/bin/activate
python igram_finder_gui.py
```

The left panel lets you add/remove directories, set all detection options, and
start or cancel a run.  The right panel lists every image with a status symbol
(`[✓]` ok, `[~]` low-fringe, `[!]` failed, `[S]` skipped) — click any entry
to see a cropped preview of the detected circle. Double click to de-select it.

---

## Usage

`igram_finder` takes a directory containing interferogram images ("igrams") in JPG format. These
igrams should have a large X or Y tilt so there are many (30-40) fringes. If not, `igram_finder`
won't work so well, and to be honest, neither will DFTFringe.

### Detect and write `.oln` files

```bash
python igram_finder.py <directory>
```

Common options:

| Option          | Default | Description     |
|-----------------|---------|-----------------|
| `--force`       |   off   | Overwrite existing `.oln` files |
| `--dry-run`     |   off   | Detect and print results without writing |
| `--compare`     |   off   | Compare against existing `.oln` files (for testing) |
| `--re-look`     |   off   | Second pass: use session consensus to fix outlier detections |
| `--scale N`     |   4     | Downsample factor (4 = quarter resolution) |
| `--threshold N` |   20    | Brightness threshold for outer-envelope detection (0–255) |
| `--jump-warn N` |   50    | Warn if circle centre moves > N px between frames |
| `--no-spots`    |   off   | Disable bright internal-reflection spot detection (on by default) |
| `--spot-threshold F` | 0.35 | Brightness excess over local background to flag a spot (lower = more sensitive) |

Example:

```bash
python igram_finder.py igrams/AT8IN/2024-08-26-3.3V-8X-low-tilt/00-North-up
python igram_finder.py igrams/200mm-f6/2023-06-27/90-CW --re-look
```

The `fringe=X.XX` value printed for each image is the fringe visibility score
(higher = more fringes = more reliable detection).  Images marked
`[low-fringe]` are near-zero tilt and may not be suitable for DFTFringe.

The `--re-look` option causes `igram_finder` to go back and re-look all the igrams it analyzed to
see if they all fall into a consistent range. The assumption is that all the igrams in the
subdirectory were obtained without moving the mirror, so the igram circle should be the same within
a couple of pixels in each image. The first image is a "cold start", while subsequent images rely on
this first estimate. Since the cold-start estimate might not be as good as later estimates, we redo
the whole stack again and make corrections if the re-estimate is significantly different. This takes
very little time.

The `--compare` option assumes there are existing `.oln` files present. For this command to be
useful, these existing `.oln` files should have been created by a person finding the igram circle
themselves. If done correctly, these outlines can be considered as "ground truth", and `--compare`
shows how well `igram_finder` has done in comparison.

By default `igram_finder` also looks for **bright internal-reflection spots** inside the mirror —
compact bright glows caused by a stray reflection somewhere in the optical system, which corrupt the
wavefront if left in. Any it finds are written as polygon `regions` in the `.oln` so DFTFringe
ignores those pixels. A spot is flagged where the fringe-suppressed brightness rises above the
smooth local background by at least `--spot-threshold` (a fraction, default `0.35` = 35 %). Bright
damaging spots sit well above this; lower the threshold to also catch fainter ones (at some risk of
false positives), or use `--no-spots` to turn the feature off. Because a reflection is fixed in the
optical system, each frame's search is seeded from the previous frame's spots, so a spot that dims
slightly between frames is still caught. Spot detection is skipped/unreliable on near-zero-tilt
`[low-fringe]` images, which DFTFringe does not process anyway.

### Evaluate accuracy across a dataset tree

```bash
python igram_eval.py [root_dir]
```

`igram_eval.py` walks `root_dir` (default: `igrams/`), finds every JPEG that
has a matching hand-placed `.oln` file, runs the detector, and reports
per-image and per-dataset accuracy.

Common options:

| Option               | Default | Description |
|----------------------|---------|-------------|
| `--verbose` / `-v`   | off     | Print per-image detection details |
| `--summary-only`     | off     | Skip per-directory listing |
| `--scale N`          | 4       | Downsample factor |
| `--threshold N`      | 20      | Brightness threshold |
| `--overexp-thresh T` | 0.10    | Flag images with ≥ T fraction of saturated pixels |

---

## Output format

Detection results are written as DFTFringe `.oln` JSON files alongside the
source JPEG:

```json
{
    "dft_filter_radius": 25,
    "inside_outline":  { "center_x": -1, "center_y": -1, "radius_x": 0, "radius_y": 0 },
    "outside_outline": { "center_x": 2847.3, "center_y": 1992.1, "radius_x": 831.7, "radius_y": 831.7 },
    "regions": [
        [ { "x": 3120, "y": 2024 }, { "x": 3172, "y": 2090 }, { "x": 3128, "y": 2092 } ]
    ]
}
```

All coordinates are in full-resolution pixels, origin at the top-left corner
(standard image convention, which DFTFringe uses).

`regions` is a list of polygons marking bright internal-reflection spots for
DFTFringe to ignore; each polygon is a list of `{ "x", "y" }` vertices. It is
empty when no spots are found (or when `--no-spots` is used).

---

## Algorithm overview

See `ALGORITHM.md` for a full explanation.  In brief:

1. **Outer envelope** — brightness threshold + largest contour gives a mask
   bounding the illuminated area.
2. **Variance mask** — local variance identifies the fringe region (fringes
   have high spatial-frequency content); Otsu threshold + morphological
   closing produces a filled mirror-shaped binary mask.
3. **RANSAC circle fit** — robust to asymmetric fringe coverage.
4. **Radial gradient refinement** — snaps the circle to the actual mirror edge
   at sub-pixel accuracy by finding the steepest brightness drop at each of
   720 radial angles.
5. **Bright-spot detection** — a fringe-suppressing blur minus a smooth
   background (large morphological opening) isolates compact bright
   internal-reflection glows; blobs whose excess exceeds `--spot-threshold` of
   the local background, after area and count sanity filters, become `regions`.

Optional fallbacks handle edge cases: outer-ring latch (Stage 3 grabs the
illumination envelope), prev-circle mismatch (Stage 4 jumps > 10 % between
frames), and the `--re-look` second pass (session consensus re-anchors
outliers within a 20–80 px window).
