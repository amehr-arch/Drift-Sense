# Deliverables index

Where each required item lives in this repository, and how to run it.

---

## Required files

| # | Requirement | File |
|---|---|---|
| 1 | Setup instructions | [`README.md`](README.md) |
| 2 | Dataset generator (standalone `.py`) | [`generate_dataset.py`](generate_dataset.py) |
| 3 | Localisation inference script (standalone `.py`) | [`locate_pattern.py`](locate_pattern.py) |
| 4 | Batch inference over a supplied test set | [`predict_dataset.py`](predict_dataset.py) |
| 5 | Evaluation harness | [`evaluate_dataset.py`](evaluate_dataset.py) |
| 6 | Dependency list | [`requirements.txt`](requirements.txt) |
| 7 | Citations for augmentation and noise choices | [`CITATIONS.md`](CITATIONS.md) |

No deep-learning model is used, so no weights file or training notebook is
included. The rationale is below.

---

## Running each component

### Inference on one pair

```bash
python locate_pattern.py reference.png search.png
#  -> 925.493 681.672
```

Two image paths in, one coordinate pair out. No editing, no configuration file, no
trained weights. `--format json` adds the correlation score, a peak-to-sidelobe
confidence, the runner-up margin, the winning rotation and scale, and algorithm
time.

### Inference on a supplied test set

```bash
python predict_dataset.py test_data/
#  -> test_data/predictions.csv
```

Pairs are matched by filename, or listed explicitly with `--manifest`. An
unreadable pair does not stop the run.

### Generating data

```bash
python generate_dataset.py --pairs 30 --out data/set --imaging --seed 42
python generate_dataset.py --architecture finfet --pairs 30 --out data/ff --imaging
python generate_dataset.py --optical --pairs 10 --out data/opt
```

Every dataset is a deterministic function of the seed. The manifest records the
seed, the geometry and every layout and capture parameter.

### Evaluation

```bash
python evaluate_dataset.py data/set --panels 5
```

Writes accuracy against tolerance, timing, per-anchor-class breakdown, the failure
taxonomy, and match panels for the worst pairs.

### Tests

```bash
python -m unittest discover -s tests -t .
#  -> Ran 506 tests ... OK
```

Standard library only. No pytest, no plugins, no configuration file.

---

## Results

30 pairs per dataset, full SEM imaging model, measured on the solvable subset:
pairs whose reference window carries a structural anchor on both axes.

| Dataset | Median error | Within 1 px |
|---|---|---|
| Development (seed 42, n=14) | 0.096 px | 100% |
| Held out (seed 90210, n=13) | 0.707 px | 61.5% |
| **Validation (seed 31337, n=26)** | **0.354 px** | **69.2%** |
| FinFET (seed 2024, n=15) | 0.548 px | 73.3% |

The validation set was generated after the current defaults were fixed and was
never used to select any parameter, at a 16-30 nm beam spot and a 35-120 dose.
It is the figure to quote.

Across all 30 development pairs including the structurally ambiguous ones, the
figure is 46.7%. Both are reported because the difference is a property of the
problem: roughly 40% of randomly placed reference windows contain no structural
landmark on one or both axes, and are therefore recoverable only to one cell pitch
by any method. See the anchor discussion in `README.md`.

**Computation time** is about 1.5 s per 1000x1000 pair with the shipped defaults,
or about 0.7 s with `arbitrate_preprocessing=False`. Single core, no GPU, image
loading excluded.

---

## Failure analysis

`driftsense/failures.py` classifies every pair into one root cause with the
supporting evidence, written out on every evaluation run as `failure_modes.csv`,
`failure_modes.json` and `failure_report.txt`.

| Mode | Development | Held out | Attributable to |
|---|---|---|---|
| correct | 15 (50.0%) | 9 (30.0%) | - |
| unanchored_axis | 13 (43.3%) | 12 (40.0%) | the problem |
| periodic_alias | 1 (3.3%) | 3 (10.0%) | the method |
| blur_limited | 0 | 1 (3.3%) | the optics |
| unexplained | 1 (3.3%) | 5 (16.7%) | unknown |

The dominant mode in both regimes is `unanchored_axis`, which is not recoverable
by a better algorithm. The mode attributable to the method, `periodic_alias`, runs
at 3 to 10 percent.

Every failure across both datasets carried a low-confidence signal. Precision is
limited: on held-out data the threshold that catches all failures also discards
two thirds of correct answers, so the measure flags results for review rather than
rejecting them automatically. Both figures are in `README.md`.

### Worked failure case

`data/*/evaluation/panels/` contains the worst pairs from any evaluation run. On
held-out pair 20 the only structure surviving in the reference window at that dose
is a single mat boundary strip; the prediction and the truth sit on different
strips and the two sampled patches are visually indistinguishable. Its runner-up
margin was 0.00086 against 0.116 on confident pairs.

---

## Why no deep learning

Localisation runs in under two seconds on one CPU core with no training
infrastructure, no weights to ship, and no framework that must load on the
evaluator's machine. Every step is inspectable: the correlation surface can be
displayed and the answer justified directly from it.

Around 40% of pairs in this problem are structurally ambiguous, with several
equally valid answers. A coordinate-regression network trained on such pairs
minimises its loss by predicting the mean of the valid positions, which is
typically a location matching nothing. Correlation instead produces several equal
peaks and exposes the ambiguity, which is what the anchor labelling and the
failure taxonomy are built on.

The generator emits labelled, training-ready data with exact ground truth, so a
learned method remains straightforward to add.

---

## Scope and limitations

- **Optical (RGB) path.** Implemented and characterised. At a 430 nm diffraction
  limit against a 70 nm bit-line pitch the cell array is not resolved, and the
  regime is aliasing-limited: the true location is the global correlation maximum
  on 2 of 10 pairs. An optical tool would navigate by die-level features requiring
  a floorplan layer this generator does not model. The optical figure is not
  comparable with the SEM figure.
- **FinFET pitch range.** The sampler starts at a 32 nm fin pitch. Published
  ground rules reach 24 nm, which at the 10 nm wide-search pixel is 2.4 pixels and
  therefore below the sampling limit.
- **Preprocessing cost.** Arbitrated per pair rather than gated, so the locator
  runs twice: about 1.5 s per pair against 0.7 s.
- **Roughness and cross-scale exactness.** With line-edge roughness enabled the
  residual between a 1 nm/px and a 10 nm/px render rises from 2e-05 to an rms of
  3.75 grey levels, which is the nanometre-scale detail a 10 nm pixel cannot
  resolve.
- **Not modelled.** Corner rounding, periphery circuitry detail, and a measured
  line-edge-roughness power spectral density.

---

## Verification

Run from a clean copy of this repository:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t .          # 506 tests
python generate_dataset.py --pairs 30 --out data/dram_sem --imaging --seed 42
python locate_pattern.py \
    data/dram_sem/pairs/pair_0027_reference.png \
    data/dram_sem/pairs/pair_0027_search.png       # -> 925.493 681.672
```

No installation step beyond pip, no configuration, no network access. Python 3.9
or later, NumPy and Pillow.
