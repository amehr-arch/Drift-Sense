# Drift-Sense

Synthetic dataset generation and navigation-error recovery for wafer-inspection
SEM imagery.

**Problem.** Given a reference image (1000x1000, 1 nm/px) and a wide-search image
(1000x1000, 10 nm/px covering 10x the field of view), report the pixel centre
`(x, y)` in the search image where the reference pattern appears, shrunk 10x. If
more than one region matches, report the one closest to the search image centre.

**Method.** FFT-based normalised cross-correlation with rotation and scale search,
uniqueness weighting, sub-pixel refinement, and per-pair arbitration between
filtered and unfiltered passes. No trained model, no weights, no GPU.

**Dependencies.** NumPy and Pillow. Python 3.9 or later.

**Architectures.** DRAM (6F2 and 8F2) and FinFET, both with a full SEM imaging
model, line-edge roughness, and an optional three-channel optical path.

---

## Quick start

```bash
git clone <repository-url>
cd drift-sense
python -m pip install -r requirements.txt

# 1. generate a 30-pair set with the SEM imaging model, seeded
python generate_dataset.py --pairs 30 --out data/dram_sem --imaging --seed 42

# 2. locate a pattern (this is the inference entry point)
python locate_pattern.py \
    data/dram_sem/pairs/pair_0027_reference.png \
    data/dram_sem/pairs/pair_0027_search.png
#    prints:  925.493 681.672
#    ground truth for this pair is 925.5 681.7, so the error is 0.029 px

# 3. evaluate the locator over the whole dataset
python evaluate_dataset.py data/dram_sem --panels 5

# 4. measure accuracy against one imaging term at a time
python sweep_noise.py --parameter spot_size_nm --values 4 14 26 40

# tests (standard library only, no pytest required)
python -m unittest discover -s tests -t .
```

The generator is seeded, so `--seed 42` reproduces the same thirty pairs on any
machine. Ground truth for every pair is in `data/dram_sem/ground_truth.csv`.

Pair 27 is used in the example because its `anchor` column reads `both`, meaning
its reference window carries a structural landmark on each axis and the answer is
uniquely determined. Roughly half of randomly placed windows are not like that: a
window landing inside uninterrupted array is periodic on one or both axes and is
recoverable only to one cell pitch by any method. See [Anchors and
solvability](#anchors-and-solvability).

---

## The inference script

`locate_pattern.py` takes two image paths and prints one coordinate pair. No
editing, no configuration file, no trained weights.

```bash
python locate_pattern.py ref.png search.png                 # 925.493 681.672
python locate_pattern.py ref.png search.png --format json   # full result
python locate_pattern.py ref.png search.png --format csv    # x,y header + row
```

JSON output for pair 27 of the seeded set above:

```json
{
  "x": 925.493, "y": 681.6723,
  "score": 0.882284,
  "confidence_psr": 6.638,
  "tie_broken": false,
  "runner_up_margin": 0.071235,
  "template_size_px": 100,
  "hypothesis": { "scale": 1.0, "rotation_deg": -0.9 },
  "n_hypotheses": 27,
  "preprocessed": true,
  "arbitration_margin": [0.071235, 0.066064],
  "elapsed_s": 1.754941
}
```

| Field | Meaning |
|---|---|
| `x`, `y` | The answer, in search-image pixels |
| `score` | Normalised cross-correlation of the winning match |
| `confidence_psr` | Peak-to-sidelobe ratio: how far the peak stands above the rest of the correlation surface |
| `runner_up_margin` | Score gap to the next distinct candidate |
| `tie_broken` | Whether the centre tie-break selected among near-equal candidates |
| `hypothesis` | The rotation and scale that won the search |
| `preprocessed` | Whether the band-passed pass produced the returned answer |
| `arbitration_margin` | Runner-up margin of the winning and rejected passes |
| `elapsed_s` | Algorithm time, excluding image loading |

`runner_up_margin` is the most useful diagnostic after the coordinates. On the
measured datasets every failure carried a small margin, though the measure has
limited precision on unseen data: see [Confidence](#confidence).

Exit codes: `0` success, `1` bad input.

---

## Running it on someone else's data

| You have | Use | You get |
|---|---|---|
| Two images | `locate_pattern.py` | One coordinate, printed |
| A folder of pairs, no answers | `predict_dataset.py` | `predictions.csv` |
| A folder of pairs with answers | `evaluate_dataset.py` | Accuracy report, failure taxonomy, match panels |

### Batch prediction

```bash
python predict_dataset.py their_test_data/
#   writes their_test_data/predictions.csv
```

Pairs are matched by filename: every `*_reference.png` is paired with the matching
`*_search.png`. For different naming, or irregular naming:

```bash
python predict_dataset.py data/ --reference-suffix _hi.tif --search-suffix _lo.tif
python predict_dataset.py --manifest pairs.csv     # columns: reference_path, search_path
```

Output columns: `pair_id, reference_path, search_path, x, y, score,
confidence_psr, runner_up_margin, preprocessed, elapsed_s`. An unreadable pair
does not stop the run; it receives an empty `x`/`y`, is listed on stderr, and the
exit code becomes 2.

### Scoring against ground truth

`evaluate_dataset.py` needs a `ground_truth.csv` beside a `pairs/` directory. Only
five columns are required:

```csv
pair_id,reference_path,search_path,gt_x,gt_y
0,pairs/ref_0.png,pairs/search_0.png,239.200,657.300
```

Paths are relative to the dataset directory. `gt_x` and `gt_y` are the match
centre in search-image pixels.

Optional columns enable more analysis. `anchor`, `anchor_x` and `anchor_y` split
solvable from ambiguous pairs; `pitch_x_nm`, `pitch_y_nm`, `feature_size_nm` and
`search_pixel_size_nm` let the failure taxonomy diagnose aliasing and blur limits.
Without them everything still runs, with anchor reported as `unknown`.

### Two input conventions to check

**Zoom ratio.** The problem statement fixes the search image at 10x the linear
field. For other ratios pass `--zoom-ratio`. If the mismatch makes the template
too large for the search image, the error names the option and prints both image
sizes.

**Coordinate convention.** This repository uses pixel `(row i, col j)` spanning
`[j, j+1) x [i, i+1)`, reported as `(x, y)` with x the column axis. Corner-based or
1-indexed ground truth must be converted before comparison, or every error carries
a constant offset. Unequal image sizes raise a warning, since the problem
statement fixes both captures at the same pixel count.

---

## How localisation works

```
reference (1000x1000, 1 nm/px)
      |  area-average reduce by the zoom ratio
      v
  template (100x100, 10 nm/px)      search (1000x1000, 10 nm/px)
      |                                     |
      +-------------> NCC via FFT <---------+
                          |
                 non-maximum suppression
                          |
                   centre tie-break
                          |
                 sub-pixel refinement
                          |
                       (x, y)
```

**Area-average reduction.** The reference is reduced by exactly the zoom ratio,
each output pixel being the mean of the corresponding 10x10 block. This matches
what a coarser beam and larger pixels physically produce. Interpolating upward
would introduce detail the search image never recorded.

**Normalised cross-correlation via FFT.** The correlation surface gives a score at
every candidate position. Normalisation makes the score invariant to brightness
and contrast differences between the two captures. Window means and variances come
from integral images, so normalisation costs constant time per position. The full
surface takes about 30 ms.

**Non-maximum suppression.** Peaks are separated by at least half a template side,
so each reported candidate describes a distinct region. The shortlist holds 16.

**Centre tie-break.** Candidates whose scores differ by less than `tie_tolerance`
(0.01 in NCC units) are treated as equally good, and the one closest to the search
image centre is selected, as the problem statement requires.

**Sub-pixel refinement.** A parabola through the peak sample and its neighbours on
each axis gives the vertex. Reference crops begin at whole nanometres, one tenth
of a search pixel, so the true answer is rarely on the integer grid.

**Rotation and scale search.** Nine rotations spanning +/-3 degrees and three
scales spanning +/-3 percent give 27 hypotheses. Each is ranked on a 4x-reduced
copy; the best three are re-scored at full resolution. The un-warped hypothesis is
always evaluated, and a warped alternative must exceed it by `hypothesis_margin`
in peak-to-sidelobe units before it is accepted.

**Uniqueness weighting.** Template pixels are weighted by how much they
discriminate, computed from the template's own autocorrelation. Repeating array
interior counts for less; mat boundaries count for more.

**Alignment refinement.** Position, rotation and scale are polished together by
coordinate descent, off the discrete search grid.

**Preprocessing arbitration.** The pipeline runs twice, once with band-pass
filtering and local contrast normalisation and once without, and returns the
answer whose runner-up margin is larger. See
[Preprocessing](#preprocessing-and-arbitration).

---

## Results

30 pairs per dataset, full SEM imaging model, realistic stage error, measured on
the **solvable subset**: pairs whose reference window carries a structural anchor
on both axes.

| Dataset | Median error | Within 1 px |
|---|---|---|
| Development (seed 42, n=14) | 0.096 px | 100.0% |
| Held out (seed 90210, n=13) | 0.707 px | 61.5% |
| **Validation (seed 31337, n=26)** | **0.354 px** | **69.2%** |
| FinFET (seed 2024, n=15) | 0.548 px | 73.3% |

The validation set was generated after the current defaults were fixed and was
never used to select any parameter. It uses a 16-30 nm beam spot and a 35-120
dose, harsher than either of the others. It is the figure to quote.

Across all 30 development pairs including the structurally ambiguous ones, the
figure is 46.7%. Both numbers are reported because the difference between them is
a property of the problem rather than of the method: see
[Anchors and solvability](#anchors-and-solvability).

### Computation time

| Configuration | Time per pair |
|---|---|
| Shipped defaults (two arbitrated passes) | about 1.5 s |
| Single pass (`arbitrate_preprocessing=False`) | about 0.7 s |
| FFT correlation alone | about 30 ms |
| Generating one pair | about 1.8 s |

Single core, no GPU. Image loading excluded, since the reported figure is
algorithm time.

### Contribution of each stage

Measured on the development set with preprocessing enabled, which was the
configuration used while the increments were being added. The shipped defaults
arbitrate preprocessing per pair, so these figures show the contribution of each
component rather than the current operating point.

| Locator | Median error | Within 1 px | Time |
|---|---|---|---|
| Correlation and tie-break only | 0.1675 px | 92.9% | 179 ms |
| Plus preprocessing, hypothesis search, gating | 0.1480 px | 92.9% | 859 ms |
| Plus uniqueness weighting | 0.1024 px | 100.0% | 859 ms |
| Plus alignment refinement | 0.0776 px | 100.0% | 927 ms |

Uniqueness weighting is what takes the solvable subset to 100%. The remaining
failure at that point is periodic alias substitution, and weighting the
discriminating pixels more heavily addresses it directly.

At 3 degrees of tilt the full locator reaches 78% within 1 px where the
correlation-only baseline manages 33%.

---

## Anchors and solvability

A DRAM array repeats every 70-190 nm and a FinFET grating every 32-54 nm. Within a
reference window that contains no structural landmark, the position along that
axis is recoverable only to one cell pitch, by any method, because the information
required to distinguish one repeat from another is not present in the two images.

The generator labels every pair with whether its reference window contains an
anchor on each axis:

| Label | Meaning |
|---|---|
| `both` | Landmark on both axes. Uniquely solvable. |
| `x` or `y` | Landmark on one axis only. The other is recoverable to one cell pitch. |
| `none` | No landmark. Position recoverable to one cell pitch on both axes. |

Evaluation reports accuracy separately for each class. Held-out performance in
full:

| Anchor class | n | Median error | Within 1 px |
|---|---|---|---|
| both axes | 13 | 0.80 px | 61.5% |
| x only | 4 | 273.4 px | 0% |
| y only | 12 | 113.5 px | 8.3% |
| neither | 1 | 197.4 px | 0% |

In DRAM the anchors are mat boundaries; in FinFET they are diffusion breaks.
Setting `mat_pitch_variation=0` restores a perfectly periodic layout in which
distinct mats are indistinguishable, retained as the controlled ambiguity case.

---

## Confidence

Two measures accompany every answer.

**Peak-to-sidelobe ratio** is how far the winning peak stands above the rest of
the correlation surface, in standard deviations. Confident pairs measure about 5.5
to 6.7; failures about 3.7 to 5.5.

**Runner-up margin** is the score gap to the next distinct candidate.

Measured separation between correct answers and failures:

| | Development | Held out |
|---|---|---|
| Failures | 15 | 21 |
| Runner-up margin, correct answers | 0.0025 - 0.1424 | 0.0023 - 0.0352 |
| Runner-up margin, failures | 0.0002 - 0.0112 | 0.0002 - 0.0187 |
| Tightest threshold catching every failure | 0.0112 | 0.0187 |
| Correct answers that threshold also discards | 2 of 15 | 6 of 9 |

Every failure across both datasets carried a low-confidence signal, so recall is
complete. Precision is limited: on held-out data the threshold that catches all
failures also discards two thirds of the correct answers. The measure is therefore
useful for flagging results that warrant review, not for automatic rejection.

---

## Failure taxonomy

`driftsense/failures.py` classifies every pair into one root cause, with the
evidence that selected it. It runs by default as part of any evaluation and writes
`failure_modes.csv`, `failure_modes.json` and `failure_report.txt`.

| Mode | Meaning | Source |
|---|---|---|
| `correct` | Within the 1 px tolerance | - |
| `unanchored_axis` | The failing axis carries no landmark; recoverable only to one cell pitch | The problem |
| `periodic_alias` | The axis carried a landmark, but the prediction is a whole number of cell pitches away | The method |
| `blur_limited` | Search beam spot at or above 0.7x the feature size; the structure is not in the image | The optics |
| `subpixel_drift` | Correct cell, error under half a pitch | The method |
| `unexplained` | None of the above | Unknown |

The modes are not disjoint in nature, so classification takes the first match in
that precedence order and the counts partition the dataset. `unanchored_axis`
precedes `periodic_alias` because the first describes an unavoidable error and
only the second is attributable to the method.

| Mode | Development | Held out | FinFET |
|---|---|---|---|
| correct | 15 (50.0%) | 9 (30.0%) | 12 (50.0%) |
| unanchored_axis | 13 (43.3%) | 12 (40.0%) | 7 (29.2%) |
| periodic_alias | 1 (3.3%) | 3 (10.0%) | 2 (8.3%) |
| blur_limited | 0 | 1 (3.3%) | 3 (12.5%) |
| unexplained | 1 (3.3%) | 5 (16.7%) | 0 |

`unanchored_axis` dominates in every regime at around 40% of pairs. Those errors
are not recoverable by a better algorithm. `periodic_alias`, which is
attributable to the method, runs at 3 to 10 percent.

Each alias verdict reports the probability that a random offset would pass the
same test, so a weak attribution on a fine pitch is visible as such. The test
requires the residual to lie within 1 px absolutely as well as within a quarter of
a pitch, because a fractional criterion alone is close to vacuous when the pitch
is only a few pixels.

`failures.py` reads layout pitches and capture settings, which the locator cannot.
It is an analysis tool run after the fact by someone holding the ground truth, and
a test asserts it never imports `locate` or `correlate`.

---

## Preprocessing and arbitration

Band-pass filtering removes structure coarser than the template, which suppresses
illumination gradients and charging, and structure finer than the coarser capture
can carry, which prevents the reference being credited with detail the search
image never recorded. Local contrast normalisation divides out slowly varying
contrast so a faint region matches as readily as a strong one.

Applied unconditionally, preprocessing improves the development regime and
degrades a harsher one:

| Locator | Development | Held out | Validation |
|---|---|---|---|
| Preprocessing off | 0.344 px, 100% | 0.802 px, 61.5% | 0.703 px, 53.8% |
| Preprocessing on | 0.078 px, 100% | 219.4 px, 38.5% | 0.358 px, 65.4% |
| **Arbitrated (shipped)** | **0.096 px, 100%** | **0.707 px, 61.5%** | **0.354 px, 69.2%** |

The shipped configuration runs both passes and returns the answer whose runner-up
margin is larger. A degraded pass leaves its peak closer to its runner-up, so the
margin ranks the two answers using only the correlation surfaces, with no
knowledge of dose, spot size or ground truth.

Arbitration is at least as good as disabling preprocessing on every metric in
every regime, and avoids the collapse seen when it is always enabled. The cost is
a second full pass, about 1.5 s per pair against 0.7 s.
`LocalisationConfig(arbitrate_preprocessing=False)` restores the single pass.

`driftsense/quality.py` provides a spectral estimate of how far real structure
survives above the noise floor, reported as a fraction of Nyquist. It falls
monotonically with beam spot, from 0.99 at a 4 nm spot to 0.12 at 40 nm. It is
available as a diagnostic and is not used to control preprocessing.

---

## The SEM imaging model

`--imaging` passes each capture through eleven physical steps, in the order the
microscope applies them: material contrast, edge brightening, beam point spread,
inter-visit stage error, charging, vignetting, shot noise, detector noise, gamma,
sensor defects, quantisation. Noise enters only after detection, because that is
where a detector introduces it.

The two captures receive independent random generators and different dose. The
wide-field image collects about a tenth of the electrons per pixel, which is the
physical reason it is noisier. The 10:1 dose ratio follows the organisers'
released sample metadata.

Two modelling decisions worth stating:

- **No oversampling is needed for the blur.** The physical model is specimen, then
  PSF, then pixel integration. The renderer already integrates over the pixel, so
  applying the PSF afterwards computes `(specimen * box) * psf` where the physics
  requires `(specimen * psf) * box`. Convolution commutes, so the two are
  identical.
- **Geometric error is applied to the reference capture.** The answer is expressed
  in search-image coordinates, so the search image defines the frame and any
  relative misalignment belongs to the other image. This keeps the ground truth
  exactly computable while presenting the locator with the full distortion.

Every term is justified against sources in [CITATIONS.md](CITATIONS.md).

---

## Degradation study

Accuracy against one imaging term at a time, on the solvable subset, everything
else held at a fixed operating point and the same scenes reused at every level.
Eight scenes per level.

| Term | Level | Median error | Within 1 px |
|---|---|---|---|
| dose (search) | 1600, 200, 50 | 0.213, 0.436, 0.489 px | 88%, 100%, 100% |
| **beam spot (nm)** | 6, 16, 30 | 0.292, 0.499, **126.1 px** | 100%, 88%, **50%** |
| rotation (deg) | 0, 1.5, 3.0 | 0.357, 0.366, 0.927 px | 88%, 88%, 62% |

**Shot noise is not the limiting term.** A 32x dose reduction leaves accuracy at
100% and costs a quarter of a pixel. Normalised cross-correlation integrates over
ten thousand template pixels, so uncorrelated noise averages down by a factor of
about a hundred.

**Beam spot is the limiting term.** Between a 16 nm and a 30 nm spot the median
error rises from half a pixel to 126, because the high-frequency content that
separates one lattice alias from its neighbour is no longer present. This is the
same term that dominates held-out failures, and it is the clearest target for
further work.

**Rotation degrades gently** under the hypothesis search, losing about a quarter
of accuracy by 3 degrees.

---

## Coordinate convention

Pixel `(row i, col j)` occupies `[j, j+1) x [i, i+1)`, so an image of size N spans
`[0, N)` and the centre of pixel `(i, j)` is at `x = j + 0.5, y = i + 0.5`.
Coordinates are reported as `(x, y)` with x the column axis.

The convention is fixed in `driftsense/geometry.py` and enforced by a regression
test against the coordinates released with the organisers' sample data: world
positions 2491 nm and 5685 nm map to search pixels 299.1 and 618.5.

Ground truth is computed arithmetically rather than estimated:

```
box_x = origin_x_nm / search_pixel_size_nm
x     = box_x + template_size_px / 2
```

---

## How a pair is built

1. Sample layout parameters (feature size, cell architecture, pitches, mat size,
   strip width, phases, per-mat variation).
2. Sample a crop placement, with a configurable probability of steering the crop
   so that a structural boundary falls inside the reference window.
3. Render the reference window at 1 nm/px and the full search field at 10 nm/px
   from the same layout model.
4. Apply line-edge roughness to both, using one world-coordinate field.
5. Apply the imaging model to each capture with independent random generators.
6. Compute ground truth arithmetically from the placement.
7. Validate the pair and record the anchor labels.

Every dataset is a deterministic function of a single integer seed, and the
manifest records the seed, the geometry and every layout parameter. One seed
sequence is spawned per pair, so pair *k* is reproducible in isolation.

---

## Layout model

### DRAM

Horizontal word lines crossed by vertical bit lines with a contact at every
intersection, organised into mats separated by periphery strips.

Cell dimensions follow the lithographic feature size `F`. `6F2` gives a 2F x 3F
cell (bit-line pitch 2F, word-line pitch 3F); `8F2` gives 2F x 4F. Defaults for
`mat_size_nm` (2600) and `strip_width_nm` (320) follow the organisers' released
sample metadata.

Each mat column carries its own bit-line pitch and each mat row its own word-line
pitch, drawn deterministically from the mat index. `mat_width_variation` varies
line width independently of pitch, representing critical-dimension
non-uniformity. Both are applied per column and per row, which keeps every layer a
product set and preserves exact separable rendering.

### FinFET

Parallel fins at the fin pitch, crossed by gate lines at the contacted poly pitch,
with source and drain contacts on the fins between gates. Blocks of standard cells
are separated by diffusion breaks, which provide the anchors.

Occlusion resolves to four disjoint product sets, so the render is area-exact at
any pixel size. Residual between a 1 nm/px render and a 10 nm/px render of the
same region: **2.1e-05 grey levels**, matching the DRAM model.

FinFET is the harder architecture under identical optics. Ground-truth correlation
runs 0.36-0.79 with a median of 0.65, against DRAM's 0.71-0.90 with a median of
0.83. Fin pitches of 32-54 nm imaged with an 8-18 nm spot at 10 nm pixels sit
closer to the resolution limit than DRAM's 48-96 nm bit-line pitch, so less of the
reference survives into the wide-search capture. The validation floor is set per
architecture accordingly.

The sampler's fin-pitch range starts at 32 nm. Published ground rules reach 24 nm,
which at the 10 nm wide-search pixel is 2.4 pixels and therefore below the
sampling limit; such pairs are unsolvable for reasons of optics rather than
algorithm.

### Exact rasterisation

Both layouts are Manhattan, so every material region is a product set and its
two-dimensional coverage field is the outer product of two one-dimensional
coverage vectors. Coverage along each axis is computed in closed form from
cumulative stripe length rather than by supersampling, which makes the render
exact and about ten times faster.

Cross-scale agreement between a 1 nm/px and a 10 nm/px render of the same region:
**2.8e-05 grey levels** for DRAM, **2.1e-05** for FinFET.

---

## Line-edge roughness

Printed edges deviate from their nominal position by a nanometre or two,
correlated along the line over tens of nanometres. `roughness.py` models this as a
displacement field applied to the rendered coverage: displacing a coverage field
moves its edges, and its uniform interiors are unaffected.

Two properties are enforced and tested.

**It is a property of the specimen, not the capture.** Both captures are displaced
by the same world-coordinate field, because both image the same silicon.
Independent per-capture roughness would remove information that is genuinely
available.

**It is anisotropic.** Roughness is correlated along a line and independent
between neighbouring lines. `dx` varies slowly along y and quickly across x, and
`dy` is the transpose. An isotropic field would displace adjacent lines together,
which is stage drift rather than roughness.

The field is a sum of sinusoids evaluated analytically at world coordinates, so
the same physical point is displaced identically at any pixel size.

| Measurement | Roughness off | Roughness on (1.2 nm) |
|---|---|---|
| Cross-scale residual, worst pixel | 2e-05 | 27.8 grey levels |
| Cross-scale residual, typical pixel | 2e-05 | 0.19 |
| Correlation at the true location | 0.833 | 0.827 |
| Median localisation error | 0.176 px | 0.260 px |
| Within 1 px | 75% | 75% |

The cross-scale residual rises because area-averaging a displaced edge is not the
same as displacing an averaged edge, and the difference is the nanometre-scale
detail a 10 nm pixel cannot resolve. Roughness is therefore applied only alongside
the imaging model; the noise-free renders remain exact. `--ler-nm 0` disables it.

---

## Optical microscope (RGB)

`optical.py` provides a three-channel diffraction-limited path, reusing the same
layout models, coordinate convention and ground truth. Blur is applied per channel
because the Rayleigh limit scales with wavelength, so blue resolves finer than
red.

At 550 nm and NA 0.9 the diffraction limit is about 430 nm, roughly six times a
DRAM bit-line pitch. The cell array is therefore not resolved at all, and only
mat-scale structure survives. This inverts the structure of the problem: under the
SEM the fine grating carries the signal and mat boundaries are the rare anchor,
while optically the grating is absent and the boundaries are the only available
signal.

Measured on 10 optical pairs at a 30 nm/px reference and 300 nm/px search:

| Measurement | Value |
|---|---|
| Correlation at the true location | 0.898 |
| Correlation at the best location | 0.938 |
| True location was the global maximum | 2 of 10 pairs |

The optical regime is aliasing-limited rather than noise-limited: the correlation
at the true location is strong, but another location scores higher. An optical
tool would navigate by die-level features tens of microns across, which requires a
floorplan layer this generator does not model. The optical figure is therefore not
comparable with the SEM figure and should not be quoted alongside it.

`locate` accepts three-channel input and reduces it to luminance. Generate optical
pairs with `--optical`.

Colour is modelled as a flat per-material reflectance. Real optical colour arises
from thin-film interference in the dielectric stack, which requires a film model
this generator does not include.

---

## Repository layout

```
driftsense/
├── geometry.py       coordinate convention, world <-> image mapping, ground truth
├── raster.py         area-exact rasterisation of separable Manhattan layouts
├── resample.py       area-average reduction, bilinear sampling
├── correlate.py      normalised cross-correlation, peaks, sub-pixel fitting
├── locate.py         the localisation pipeline
├── evaluate.py       accuracy-versus-tolerance harness
├── preprocess.py     band-pass, local contrast normalisation, box statistics
├── quality.py        spectral resolution estimate (diagnostic)
├── failures.py       root-cause taxonomy for every result
├── imaging.py        SEM image formation: yield, edges, PSF, noise, artefacts
├── optical.py        three-channel optical image formation
├── roughness.py      line-edge roughness as a world-coordinate displacement field
├── sweep.py          degradation studies, one imaging term at a time
├── sampling.py       randomised layout parameters and crop placements
├── generate.py       pair and dataset generation
├── validate.py       invariant checks and independent geometry proof
├── visualise.py      verification panels, match panels, accuracy curves
└── layouts/
    ├── base.py       LayoutModel interface
    ├── dram.py       DRAM-style implementation
    └── finfet.py     FinFET-style implementation
generate_dataset.py   dataset generation entry point
locate_pattern.py     inference entry point (run this on test data)
predict_dataset.py    batch inference over a folder, no ground truth needed
evaluate_dataset.py   evaluation entry point
sweep_noise.py        degradation study entry point
CITATIONS.md          justification of every augmentation and noise choice
tests/                unittest suite (506 tests)
```

`locate.py` imports `geometry` and `resample` only, never a layout module. The
locator cannot consult the model that produced the data, and a test enforces this.

---

## Validation

Every generated pair is checked before it is written:

- The block-reduced reference correlates with the search image at the ground-truth
  location above a threshold (0.95 for noise-free renders, 0.55 for SEM imagery,
  0.30 for FinFET SEM imagery).
- Both images carry enough contrast to be usable.
- Dimensions match the requested geometry.

A failing pair aborts generation rather than being written.

The suite also verifies that the same world region rendered at 1 nm/px and at
10 nm/px agrees after area-average reduction, which is what makes the ground truth
trustworthy across the two scales.

---

## Reproducibility

Every dataset is a deterministic function of one integer seed. The manifest
records the seed, the geometry, and every layout and capture parameter. Running
the same command on another machine reproduces the same pairs and the same
numbers.

```bash
python -m unittest discover -s tests -t .    # 506 tests
```
