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

| Script            | Purpose |
|-------------------|---------|
| `igram_finder.py` | Detect mirror aperture and write `.oln` files for a directory of JPEGs |
| `igram_eval.py`   | Walk a directory tree, compare detections against hand-placed ground-truth `.oln` files, and report accuracy |

---

## Requirements

- Python 3.10 or later
- [OpenCV](https://pypi.org/project/opencv-python/) (`cv2`)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/) (for `scipy.ndimage.uniform_filter`)

---

## Setup: creating a virtual environment

```bash
# 1. Create the venv (once)
python3 -m venv .venv

# 2. Activate it
#    Linux / macOS:
source .venv/bin/activate
#    Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install opencv-python numpy scipy
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

| Option          | Default | Description |
|-----------------|-----|-----------------|
| `--force`       | off | Overwrite existing `.oln` files |
| `--dry-run`     | off | Detect and print results without writing |
| `--compare`     | off | Compare against existing `.oln` files (for testing) |
| `--re-look`     | off | Second pass: use session consensus to fix outlier detections |
| `--scale N`     | 4   | Downsample factor (4 = quarter resolution) |
| `--threshold N` | 20  | Brightness threshold for outer-envelope detection (0–255) |
| `--jump-warn N` | 50  | Warn if circle centre moves > N px between frames |

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

### Evaluate accuracy across a dataset tree

```bash
python igram_eval.py [root_dir]
```

`igram_eval.py` walks `root_dir` (default: `igrams/`), finds every JPEG that
has a matching hand-placed `.oln` file, runs the detector, and reports
per-image and per-dataset accuracy.

Common options:

| Option | Default | Description |
|---|---|---|
| `--verbose` / `-v` | off | Print per-image detection details |
| `--summary-only` | off | Skip per-directory listing |
| `--scale N` | 4 | Downsample factor |
| `--threshold N` | 20 | Brightness threshold |
| `--overexp-thresh T` | 0.10 | Flag images with ≥ T fraction of saturated pixels |

---

## Output format

Detection results are written as DFTFringe `.oln` JSON files alongside the
source JPEG:

```json
{
    "dft_filter_radius": 25,
    "inside_outline":  { "center_x": -1, "center_y": -1, "radius_x": 0, "radius_y": 0 },
    "outside_outline": { "center_x": 2847.3, "center_y": 1992.1, "radius_x": 831.7, "radius_y": 831.7 },
    "regions": []
}
```

All coordinates are in full-resolution pixels, origin at the top-left corner
(standard image convention, which DFTFringe uses).

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

Optional fallbacks handle edge cases: outer-ring latch (Stage 3 grabs the
illumination envelope), prev-circle mismatch (Stage 4 jumps > 10 % between
frames), and the `--re-look` second pass (session consensus re-anchors
outliers within a 20–80 px window).
