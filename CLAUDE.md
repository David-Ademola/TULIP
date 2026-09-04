# TULIP — project context

Read this first. It records decisions and measurements that are expensive to
rediscover, and flags several traps that have already cost time once.

---

## 1. What this project actually is

The repo is named "MAMMO Implementation" but the product is **TULIP** —
*Triage, Urgency, and Longitudinal Intelligent Prediction* — an AI mammography
triage and diagnostic-assistance platform for private hospitals in Lagos and
Abuja, submitted to the OPay Innovation Challenge 2026 by a five-person
University of Lagos team. Ademola Akinwande is the ML engineer; the others
cover React frontend, ASP.NET Core backend, AWS and DevOps.

**MAMMO** (Kyono, Gilbert & van der Schaar, 2018 — "A Deep Learning Solution
for Facilitating Radiologist-Machine Collaboration in Breast Cancer Diagnosis")
is being reproduced **as a baseline only**. The paper PDF has been shared into
the working sessions; cite it by section (e.g. "Appendix C-B") when relevant.

Current scope is **T and U**. The Longitudinal piece is deferred.

Two product modules:

1. **Triage Engine** — scores every queued case on an *Abnormality Index*
   (composite of lesion characteristics, morphology, BI-RADS, tissue density)
   and **ranks the radiologist's queue**.
2. **Diagnostic Co-pilot** — auto-drafts structured reports for high-confidence
   cases, escalates ambiguous ones for manual review.

### The phase mapping is not one-to-one with the paper

This matters and is easy to get wrong:

| MAMMO stage | TULIP role |
| --- | --- |
| CNN (Phase 1) | per-view feature extractor — **this is what is being built now** |
| Classifier (Phase 2) | Abnormality Index scorer, 4-view aggregation |
| Triage (Phase 3) | **Diagnostic Co-pilot** complexity gate |
| — | **Triage Engine (queue ranking)** — no MAMMO equivalent, must be built |

MAMMO's "triage" is a binary *can the AI handle this alone?* gate optimising
workload reduction. TULIP's "Triage Engine" is a **ranking** problem. They are
different objectives. MAMMO's triage network maps onto TULIP's Co-pilot, and
the headline queue-ranking feature is the one piece the baseline does not
supply.

Consequence: the auxiliary heads are **product output**, not just regularisers.
`suspicion` and `density` appear in the drafted report and feed the Abnormality
Index, so they cannot be disabled for convenience.

### Urgency labels

There is currently **no ground truth for urgency**. It is derived from BI-RADS
as a proxy. A partnership with **MAI Lab (Nigeria)** is expected to supply
radiologist-labelled urgency scores. Build the urgency scorer with a
**swappable target** so the derived index becomes a pretraining signal and the
real labels the fine-tuning target. Known limitation: BI-RADS is an *assessment*
category (suspicion of malignancy), not a measure of how fast a patient must be
seen.

---

## 2. Status: run 1 complete (2026-09-02)

Phase 1 has trained end to end on the full 20,000-image archive. Run 1
(`phase1-1280x1024-alpha-one_minus_positive_rate`): batch 48 x 2 accumulation,
head LR 3e-4, backbone 3e-5. Early stopping fired at epoch 21; best checkpoint
is **epoch 14**, `models/mammo_cnn_phase1.pt`.

Validation at epoch 14 (1,760 images, **only 84 diagnosis positives**):

| head | result |
| --- | --- |
| diagnosis | AUROC 0.805, AP 0.467 |
| ordinal (CORAL cutpoint 2) | **AUROC 0.817, AP 0.478** |
| findings | macroAP 0.100, microAP 0.236, macroAUC 0.744 |
| Mass | AP 0.313 (5.25x prevalence) |
| Suspicious Calcification | AP 0.316 (13.9x prevalence) |
| density | acc 0.768 against a 0.742 majority share -> ~2.6 pp of real signal. Accuracy is **no longer reported**; see §6 |
| age | MAE 5.78 years |

⚠️ **Do not quote AP 0.467 as "the result".** It is the maximum over 21 epochs
on the same split used to select it, and it sits **4.34 sd** above the mean of
epochs 15-21 (0.362 +/- 0.024). Bootstrapped 95% CI on that split is
**[0.368, 0.575]**, width 0.208 — 84 positives is not many. A clean estimate is
likely 0.36-0.40.

**The 12% test split is untouched.** Spend it once, on the final model, after
the alpha experiment. Every number above is a model-selection estimate.

⚠️ **The paper comparison is not apples-to-apples.** The paper's Tommy dataset
is ~20% malignant against VinDr's 4.77%, and AP's baseline *is* the prevalence,
so raw AP cannot be compared across a 4x prevalence gap. Lift over prevalence is
fairer (ours 9.8x vs the paper's 2.0x). AUROC is prevalence-independent and so
the more defensible of the two, but 0.805 vs 0.787 is still inside sampling
noise and still cross-dataset.

**It overfits after epoch 14** — train loss fell 14% (0.608 -> 0.521) while val
loss rose 15% (0.715 -> 0.824) and AP never returned within 0.07 of the peak.
Early stopping was correct; more epochs will not help. The train-loss column
falling is what "still improving" looks like, and it is misleading.

Run 1 also logged per-level suspicion AUROC, which read **below chance for
BI-RADS 2, 3 and 4** (0.456 / 0.459 / 0.433) while the >=4 cutpoint read 0.817.
That metric has since been **removed** — it measures cutpoint calibration, not
discrimination (§6). Judge the ordinal head by `suspicion_auc_per_cutpoint`.

## 3. Repo layout

```
src/model.py       MammoCNN — InceptionResNetV2 backbone + 5 heads
src/modules.py     CoralHead — rank-consistent ordinal head
src/utils.py       losses, metrics, dataset, train loop, weight helpers
src/dicom.py       DICOM -> 16-bit PNG (VOI LUT, MONOCHROME1, padding)
scripts/find_batch_size.py   doubling + bisect batch-size probe (needs CUDA)
scripts/lr_range_test.py     Smith LR range test
metadata_processing.ipynb    DICOM metadata -> breast_metadata.parquet
mammogram_processing.ipynb   DICOM -> 16-bit PNG conversion + validation
main.ipynb                   splits, weights, calibration, training
main.py                      placeholder, unused
```

⚠️ **`vindr-mammo/` is gitignored** (338 GB of DICOM + 21 GB of converted
PNG). Neither the images nor `breast_metadata.parquet` transfer via git. On a fresh machine the parquet must
be regenerated by running `metadata_processing.ipynb` against
`finding_annotations.csv` + `metadata.csv`.

---

## 4. Dataset facts (all measured, not assumed)

VinDr-Mammo: 5,000 studies × 4 views = **20,000 images**.

**Class balance — the number that drives most design decisions:**

- CNN training split: 10,556 images, **4.95% diagnosis-positive**
- `diagnosis` is defined as `breast_birads >= 4` — a *deterministic* function
  of suspicion, not a correlated label. This matters constantly.

**BI-RADS distribution (full dataset):** 1→13406, 2→4676, 3→930, 4→762, 5→226

**Density distribution:** 1→100, 2→1908, 3→**15292**, 4→2700
Density is ~76% class C with ~0.5% class A. The head cannot realistically learn
the tails; it is a report field, not an urgency driver. Do not over-invest.

**Findings — 2,226 positive labels across 1,768 images (8.8% of images):**

| class | positives | prevalence |
| --- | --- | --- |
| Mass | 1113 | 5.28% |
| Suspicious Calcification | 442 | 2.44% |
| Focal Asymmetry | 268 | 1.38% |
| Architectural Distortion | 119 | 0.61% |
| Asymmetry | 96 | 0.50% |
| Skin Thickening | 55 | 0.30% |
| Suspicious Lymph Node | 53 | 0.31% |
| Nipple Retraction | 37 | 0.21% |
| Global Asymmetry | 26 | 0.17% |
| Skin Retraction | 17 | **0.095%** |

55× prevalence range. Always report findings **per class**; a macro average
hides that the bottom four are near-unlearnable.

**`finding_annotations.csv` has 20,486 rows for 20,000 images.** Multiple rows
per image = multiple lesions. 338 images have >1 row; 252 carry genuinely
different finding categories. The metadata notebook groups by `image_id` and
unions the category lists — do **not** revert to `drop_duplicates`, which
silently discarded findings for 252 images.

**`xmin/ymin/xmax/ymax` are LESION bounding boxes, not breast.** Evidence: only
2,254 rows have boxes and all 18,232 boxless rows are `['No Finding']`; median
box is 228×244 px on a 3518×2800 image = 0.61% of area. `finding_birads` is
per-lesion, distinct from the per-breast `breast_birads` used for training.

> **Unexploited opportunity:** the concept note promises GradCAM overlays.
> These 2,254 radiologist-drawn boxes allow *quantitative* validation of those
> heatmaps (pointing-game accuracy / IoU). "Explanations validated against
> radiologist annotations" is a much stronger claim than "we produce heatmaps",
> and almost nobody does it.

**Image dimensions: 58 distinct shapes**, aspect ratios 1.256–1.398.
15,475 are 3518×2800; 3,886 are 2812×2012.

**Stored PNGs are now 1920×1536, 16-bit** — all 20,000 converted into
`vindr-mammo/images_png/`, 21 GB, ~40k distinct grey levels each, bit-exact
roundtrip. `image_path` is a column in the parquet. Regenerate with
`mammogram_processing.ipynb`; the logic lives in `src/dicom.py`.

**⚠️ Scanner is confounded with the label.** Malignancy rate by manufacturer
(n=3,000 sampled headers):

| scanner | share | diagnosis-positive |
| --- | --- | --- |
| IMS s.r.l. | 4% | **23.7%** |
| IMS GIOTTO | 1% | 11.4% |
| Planmed | 19% | 3.4% |
| SIEMENS | 76% | 3.8% |

A **6.2x** swing against an overall 4.60%, and scanner is trivially readable off
the image: SIEMENS burns a **text label** into the corner (literally `R-MLO`,
`L-CC`, saturated, fixed band at y~140-181; `BurnedInAnnotation` is absent from
the header, so metadata will not reveal it), and source dimensions differ by
vendor so the amount of padding is a second fingerprint. `main.ipynb` stratifies
splits only on `patient_cancer`, so scanner mix can drift between train and
test. Worth a GradCAM check: if heatmaps light up the corner label, the shortcut
is being used.

**73-85% of every stored image is empty background.** Tissue contrast survives
per-image standardisation intact (tissue std / whole-image std = 1.0-1.2), but
GAP averages the lesion signal across a mostly-empty feature map, and at
1280×1024 the median lesion covers ~2×2 of ~800 cells. This is the standing
ceiling on AP and the first thing to attack after the alpha experiment
(breast-region cropping, or attention pooling instead of GAP).

---

## 5. Architecture

`MammoCNN` (`src/model.py`): InceptionResNetV2 backbone (timm, ImageNet
weights, 1536-D after GAP) → Dropout(0.2) → Linear(1536, 1024) → ReLU → five
heads. 55.9 M parameters total; 1.6 M with the backbone frozen.

Forward returns a dict:

| key | shape | notes |
| --- | --- | --- |
| `diagnosis` | (B, 1) | **must stay (B,1)** — MONAI indexes `input.shape[1]` |
| `findings` | (B, 10) | multilabel sigmoid |
| `suspicion` | (B, 4) | CORAL **cutpoint** logits, `sigmoid(k) = P(y > k)` |
| `density` | (B, 4) | softmax |
| `age` | (B,) | regression |
| `diagnosis_ordinal` | (B, 1) | zero-parameter slice `suspicion[:, 2:3]` |

### CORAL suspicion head — why, and what it buys

BI-RADS is **ordered**, so a softmax is wrong: predicting 2 for a true 5 must
cost more than predicting 4, and one-hot targets make those equidistant.

CORAL (Cao, Mirjalili & Raschka 2020) converts one 5-class ordered problem into
4 binary "is y > k?" questions. A single true label answers all four at once
(`levels_from_labels`), so ordinal distance is penalised for free — the number
of violated cutpoints *is* the distance. **Measured: 6.01 loss per level of
error, exactly linear.** Predicting level 0 for true level 3 costs 18.01;
predicting 2 costs 6.01.

Rank consistency comes from sharing one weight vector across all cutpoints and
varying only the biases, so cutpoint ordering is input-independent. This
implementation goes further: biases are parameterised as
`b₀, b₀ − softplus(δ₁), …`, strictly decreasing, which **guarantees a monotone
CDF for every input**. Verified to hold across 2,000 random inputs with
deliberately hostile δ values.

Three payoffs:

1. `P(BI-RADS > 3)` is a head output — exactly the malignancy probability the
   Abnormality Index needs, monotone and ranking-ready.
2. `ordinal_class_probs()` can difference the CDF into per-level probabilities
   (needed for the paper's Table I per-level AUROC) — only valid because
   monotonicity is guaranteed.
3. `diagnosis_ordinal` lets the dedicated diagnosis head and the CORAL cutpoint
   be compared on identical labels. **RESOLVED in run 1 — the ordinal cutpoint
   wins.** Paired bootstrap on the epoch-14 checkpoint:

   ```
   ordinal - diagnosis AP  +0.0109   95% CI [+0.0044, +0.0196]
                                     P(ordinal better) = 1.000
   ```

   and it won on both AUROC and AP in **8 of 8** logged epochs (sign test
   p = 0.004). A *paired* test cancels the split noise that makes the marginal
   CIs 0.2 wide, which is why this is significant where the cross-dataset paper
   comparison is not.

   **Use `diagnosis_ordinal` as the diagnosis score at inference.** It is a
   zero-parameter slice, so this is free.

   ⚠️ That does **not** license deleting the `diagnosis` head. Its focal term
   carries `LOSS_WEIGHTS` 0.50 and 61-64% of the measured gradient share, so it
   is shaping the shared trunk the ordinal cutpoint reads from. The result shows
   it is the worse *readout*, not that the representation survives without its
   *loss*. Removing head + loss is a separate, untested ablation.

   ⚠️ Also do not confuse `diagnosis_ordinal` with the `suspicion` CORAL head
   itself. The head has real parameters, is the better predictor, and is product
   output (§1) — it cannot be removed.

---

## 6. Non-obvious decisions — read before changing any of these

### No WeightedRandomSampler. Class weighting in the loss instead

A sampler reweights whole *examples*, so with five heads it balances one label
distribution and distorts the rest. Because `diagnosis == (breast_birads >= 4)`,
oversampling on diagnosis **is** a BI-RADS resampler. Measured on the CNN split:

```
BI-RADS 1/2/3  ->  0.53x        BI-RADS 4/5  ->  10.12x     (~19x swing)
Suspicious Calcification: 2.4% -> 21.4%
```

It also destroys calibration, which TULIP cannot afford — the Abnormality Index
is a ranking score and the Co-pilot gate is a confidence threshold, so the
sigmoid must estimate P(malignant | image) at the **true** prevalence.

The paper independently reaches the same conclusion (Appendix C-C-e): class
weighting beat manual balancing because batch-16 usually contained a positive.
Their data was ~20% malignant; ours is 4.95%, so the same argument requires a
larger effective batch — hence gradient accumulation.

### Loss scale calibration — `calibrate_loss_scales()`

`LOSS_WEIGHTS` is `diagnosis 0.50, findings 0.15, suspicion 0.15, density 0.10,
age 0.10`, matching the paper's constraint that diagnosis ≥ the sum of the
auxiliaries.

But a coefficient only equals a share of the gradient if the raw losses share a
scale, and they do not — focal loss shrinks toward zero by design while CORAL
sums weighted BCE over 4 cutpoints. **Measured uncorrected:**

```
diagnosis 4%   findings 2%   suspicion 82%   density 12%   age 0%
```

The paper anticipated this: "loss weighting was adjusted **according to the
auxiliary output losses**" (Appendix C-B). Accounting for scale is faithful to
the paper, not a deviation.

`calibrate_loss_scales()` measures each task's magnitude once on the fresh model
and freezes `1/magnitude`. **Measured after: exactly 50.0 / 15.0 / 15.0 / 10.0 /
10.0, 0.00 pp deviation.**

Deliberately **not** a running average — re-normalising every step would keep
amplifying a converged task so no auxiliary could ever finish. Drift across
epochs is expected and correct.

There is a `max_scale=100.0` cap: a task whose loss starts near zero would
otherwise get an unbounded multiplier and dominate on noise (age is the
realistic candidate). It warns when it fires.

### ⚠️ The `alpha` trap — paper and MONAI use the same symbol for different things

The paper's Eq. 8 defines α as "the **inverse class frequency** tuning
parameter" and uses α = 2. MONAI implements Lin et al.'s *other*
parameterisation:

```python
alpha_factor = target * alpha + (1 - target) * (1 - alpha)   # alpha in [0, 1]
```

Passing the paper's α = 2 to MONAI gives negatives a weight of **−1.0**, which
rewards confidently wrong predictions. `compute_multi_task_loss` now raises on
`alpha ∉ [0,1]` with this explanation.

⚠️ **`weight` is silently ignored on the diagnosis head.** The obvious
workaround — translate the paper's α through MONAI's per-class `weight` instead
of `alpha` — does not work, because `focal_loss.py` guards it with
`if self.class_weight is not None and num_of_classes != 1:` and the diagnosis
head is `(B, 1)`, i.e. one class. Measured per-sample loss with logits at 0:

```
weight=None      -> [0.17329, 0.17329]   pos/neg 1.0000
weight=[0.0495]  -> [0.17329, 0.17329]   pos/neg 1.0000   <- no effect
weight=[20.0]    -> [0.17329, 0.17329]   pos/neg 1.0000   <- no effect
alpha=0.9505     -> [0.16471, 0.00858]   pos/neg 19.2020
```

No error, no warning. `findings_weights` *does* apply, because that head has 10
channels. So on a binary head `alpha` is the **only** working balance control in
MONAI, and `alpha = 1 - POSITIVE_RATE` already gives pos:neg = 19.2, which *is*
inverse-frequency weighting. Implementing the paper's α faithfully on diagnosis
would need a 2-channel head, `BCEWithLogitsLoss(pos_weight=...)`, or a manual
multiplier — not `weight`.

**Currently `ALPHA = POSITIVE_RATE = 0.0495`**, a deliberate choice by the user,
giving malignant weight 0.0495 and benign 0.9505 (pos:neg = 0.052). This follows
Lin et al., who found α < 0.5 best for the rare class at γ=2 because γ already
suppresses easy negatives hard. The alternative — `ALPHA = 1 − POSITIVE_RATE`
(inverse-frequency, pos:neg = 19.2) — is commented out one line below in
`main.ipynb`. **These differ by ~370× in relative weight on malignancy and the
comparison is an open experiment.** W&B logs `alpha_strategy` so the two runs
sort side by side.

⚠️ **`ALPHA = 1 - POSITIVE_RATE` makes the focal loss exactly class-balanced**,
so the diagnosis sigmoid is uncalibrated by construction:

```
positive weight  alpha * p       = 0.9505 * 0.0495 = 0.04705
negative weight  (1-alpha)*(1-p) = 0.0495 * 0.9505 = 0.04705   <- identical
```

The optimal input-independent prediction is therefore exactly **0.5**, confirmed
numerically (fitting a single constant logit gives 0.5000 for `1 - p` and 0.1010
for `p`, against a true prevalence of 0.0495). Run 1 used `1 - POSITIVE_RATE`,
and its checkpoint's mean prediction is 0.5165 — that is the *correct* answer
under this alpha, **not** a head failing to learn the prior. Ranking metrics are
unaffected, but §1's Abnormality Index and the Co-pilot's confidence gate both
need a real probability, so run 1 requires post-hoc recalibration (Platt or
isotonic, fitted on val) before either can use the sigmoid.

### Weight helper gotcha: normalise before clipping

`get_findings_weights` normalises to mean 1.0 **then** clips. Clipping raw
`1/prevalence` first collapses every class rarer than `1/max_weight` onto the
same ceiling — with `max_weight=20` that flattened 9 of 10 classes to ~1.0 and
silently removed all reweighting. Default is `sqrt_inverse`, because full
inverse drives Mass (most common, most clinically load-bearing) to 0.06×.
Current spread: 0.27× (Mass) to 2.02× (Skin Retraction).

### LDS kernel auto-shrink

`get_lds_weights` shrinks `kernel_size` when it exceeds the label support. The
default 5 is degenerate on density's 4 bins — it smooths toward uniform and
discards the weighting. Pass `kernel_size=3` for density explicitly.

### Training regime — deviates from the paper, deliberately

The paper cycles between training top layers and conv layers. Its stated reason
is **preserving pretrained ImageNet features** (Appendix C-B, following Shen
2017), **not** a memory constraint — they discuss memory limits explicitly
elsewhere when they hit them.

The current plan trains the **whole network from step 1**. The risk the cycling
addressed is real, so it is handled by the modern equivalent instead:

- `warmup_epochs=1.0` — linear LR ramp applied **per optimizer step**, not per
  epoch. The damage from a too-large LR on pretrained weights happens in the
  first few hundred steps.
- `backbone_lr_mult=0.1` — the backbone gets its own param group at 1/10 the
  head LR.

The plateau scheduler is guarded against stepping during warmup, or it would
read the ramp as a plateau and cut an LR that has not reached target.
`mammo_cnn.freeze_backbone()` still exists to reproduce the paper's regime.

### Learning rate 3e-4 — and why the LR range test cannot find it

**Use head LR 3e-4, backbone 3e-5.** Do not trust `scripts/lr_range_test.py`'s
number without reading its warnings; run 1's predecessor trained 13 epochs at
**2.2e-5** and reached AUROC 0.548 because the script recommended it.

Two independent defects produced that, both now fixed:

1. **Steepest-descent-of-loss was confounded by the initialisation transient.**
   The loss falls steeply at the start regardless of LR, because randomly
   initialised heads are learning the class priors. `np.gradient` divides by the
   per-step change in `log10(lr)`, so sampling *more* steps shrinks the divisor
   and inflates that transient. Measured, same data and model, only `--steps`
   changed: `--steps 30` -> 3.2e-04, `--steps 200` -> 4.5e-05. An estimator
   whose answer tracks the sampling density is measuring the sampling.
2. **`clip_grad_norm_(..., 1.0)` was on during the sweep**, bounding every
   update so a far-too-large LR could not blow up. `--clip-grad` now defaults
   to 0.

**But the deeper problem is that the loss is the wrong signal for this model.**
At batch 48 with a clean curve (signal/noise 7.76), the loss descends
0.860 -> 0.313 and then goes **flat at 0.32-0.36 all the way to LR 0.094**. At
that LR the network is destroyed, yet the loss stays far below its starting
value — because every head can collapse to its class prior, and prior-prediction
scores *better* than random init on a bounded multi-task loss. So the curve's
minimum marks where collapse-to-prior happens fastest, not where learning is
best. No estimator fixes that.

3e-4 came from measuring what actually matters instead:

- an overfit probe on 8 images: 3e-4 reaches train AP 1.000 by step 40, while
  2.2e-5 manages 0.830 in 60
- a 120-step probe at 1e-4 / 3e-4 / 1e-3 / 3e-3, scoring **validation AP**: at
  1e-3 the diagnosis-sigmoid spread *collapses* over training (0.106 -> 0.057)
  and at 3e-3 it reaches 0.030 with AUROC 0.464 — those LRs crush the head to a
  constant. 3e-4 holds the widest spread (0.169).

`scripts/lr_range_test.py` now reports a divergence-anchored suggestion, refuses
to give a confident number when the curve is unreadable (it prints
`UNRELIABLE`), and warns when steepest-descent lands inside the transient. A
signal/noise line flags a too-small batch: at batch 4 the raw loss swings
0.3 -> 2.4 between adjacent steps and nothing is measurable.

**Judge an LR by validation AP and by whether the prediction spread collapses,
never by the training loss.**

### ⚠️ Height/width conventions disagree between libraries

This transposition has bitten twice. The value is right in both places below and
**both inline comments are wrong** — do not "fix" one to match the other:

- `cv2.resize(img, dsize)` takes **(width, height)**. `preprocess_image`'s
  `target_size=(1024, 1280)` therefore yields a 1280x1024 portrait tensor.
- `torchvision` `size=` takes **(height, width)**. `IMAGE_SIZE = [1280, 1024]`
  therefore also yields 1280x1024.
- `numpy`'s `.shape` and `src/dicom.py`'s `STORE_SIZE` are **(height, width)**.

The original `target_size=(416, 320)` passed to `cv2.resize` produced a
**320x416 landscape** tensor and squashed portrait mammograms by **1.62x**.

Related trap: `RandomResizedCrop(ratio=...)` is the **absolute crop aspect
(w/h)**, not a "do not distort" flag. `ratio=(1.0, 1.0)` takes a *square* crop
and resizes it to the non-square output — measured with a test circle, every
training image was stretched **1.250x** vertically while val/test (no transform)
were not, i.e. a train/val geometry mismatch. The correct value is
`ratio = (IMAGE_SIZE[1] / IMAGE_SIZE[0],) * 2`, which measured 1.001.

### Metric choices for the ordinal heads — and one metric that lies

Suspicion (BI-RADS 1-5) and density (A-D) are both **ordinal**, so neither MAE
nor accuracy nor plain macro-F1 is the right single summary.

**Reported for both: quadratic weighted kappa (QWK).** It is the standard for
ordinal medical grading, is chance-corrected so imbalance cannot inflate it, and
penalises by *squared* ordinal distance — predicting 2 for a true 5 costs 9x
what predicting 4 costs. Precision/recall/F1 treat every misclassification as
equally wrong, which is the exact property rejected when CORAL was chosen over a
softmax, so QWK carries the ordinal information those metrics discard.

**Density accuracy is not reported, on purpose.** At a 74-76% majority share,
"always answer C" scores ~0.75. Run 1 measured 0.768 against a 0.742 baseline —
2.6 pp of real signal. Macro recall *is* balanced accuracy and is the
like-for-like replacement. Per-class F1 is also reported because with only 4
classes a macro average hides which one collapsed (class A, ~0.5% prevalence,
is expected near zero).

⚠️ **Use `average="macro"`, never `"weighted"`.** For single-label multiclass,
weighted recall is *mathematically identical to accuracy*
(`sum_k (n_k/N)(TP_k/n_k) = sum_k TP_k/N`), so switching to weighted silently
reinstates the metric that was removed. Measured on the real density marginals
with an always-predict-C model:

```
                    accuracy   macro P/R/F1              weighted P/R/F1
always predict C     0.7653    0.191 / 0.250 / 0.217     0.586 / 0.765 / 0.664
```

Weighted F1 of 0.66 makes a model that learned nothing look competent; macro
correctly reports 0.22 against a 0.25 chance level.

⚠️ **Per-level suspicion AUROC was removed — do not add it back.** `P(y=k)`
from CDF differencing is unimodal in the shared CORAL scalar, so ranking by it
means ranking by *closeness to a peak* whose position the cutpoint biases set,
not by the ordinal score. Measured on synthetic data with a **perfect** ranker:

```
                                   per-cutpoint (y>k)   per-level (y==k)
calibrated cutpoints                1.000 x4            1.000 / 1.000 / 1.000 / 1.000 / 1.000
cutpoints shifted (pos_weight-size) 1.000 x4            1.000 / 0.991 / 0.445 / 0.024 / 1.000
```

Per-cutpoint AUROC is rank-based on a single scalar and is **immune** to the
shift; per-level AUROC collapses below chance in the middle. Our
`suspicion_weights` are exactly such a shift (`[2.0, 9.53, 19.22, 20.0]`), and
run 1's shape — extremes fine, middle collapsed — matched the simulation. So
**`suspicion_auc_per_cutpoint`** ("is y > k?") is the per-threshold metric.

`ordinal_class_probs()` still exists for the Co-pilot's drafted report, which
needs a per-level BI-RADS distribution; it is simply no longer a metric. Note
the paper's Table I per-level numbers came from a *softmax* head, whose
per-class probabilities are unconstrained — that row is not reproducible with a
CORAL head, and reproducing it is not a project goal.

### One metric drives everything

`monitor` + `monitor_mode` control checkpointing, early stopping **and** LR
decay together. Previously the scheduler used `val_loss` while checkpointing
used `diagnosis_ap`; with five heads, `val_loss` is a composite that auxiliaries
can keep pushing down while the primary task has plateaued, so the LR would
never decay when diagnosis needed it.

Use **AP, not AUROC**, as the primary metric — at 4.95% prevalence AUROC
flatters. The paper makes the same argument for AUPRC (§V-B).

`patience` must exceed `lr_patience` or the run stops before a decayed LR has
any epochs to prove itself; `train()` warns if `patience <= lr_patience`.
Recommended `patience >= 2 * lr_patience + 1`.

### Both augmentation flips are intentional

`preprocess_image` canonicalises right breasts to face left, then
`RandomHorizontalFlip` and `RandomVerticalFlip` undo that. This looks wrong but
follows the paper (Appendix C-C-a): flips gave "nearly 2% AUROC" and, since
rotation up to 20° already breaks canonical orientation, full orientation
invariance pushes the network onto tumour *appearance* rather than position.
**Do not "fix" this.**

### Checkpoint format

`torch.save` now writes a dict, not a bare state_dict:
`{model, optimizer, scheduler, epoch, score, monitor, loss_scales}`.
Load with `torch.load(path)["model"]`.

---

## 7. Verified vs. unverified

**Verified by execution:**

- CORAL monotonicity holds for all inputs, including adversarial biases
- CORAL loss is exactly linear in ordinal distance (6.01/level)
- Loss scale calibration hits 50/15/15/10/10 with 0.00 pp deviation
- Warmup ramps correctly; backbone holds at exactly 0.1×; plateau decay fires
- Early stopping + resumable checkpoint (all 7 keys present)
- Metadata notebook regenerates 2,226 findings positives
- CLAHE flattens zero-padding to a near-constant (0.0022 on the 16-bit path)
- **A full 21-epoch training run on real 16-bit data** (§2)
- **AMP / `GradScaler`**: stable in the real configuration — 0/10 steps skipped,
  scale holds at 65536, loss falls. `calibrate_loss_scales` turns out to be
  load-bearing for this, not just for gradient balance: unscaled, the age MSE
  (~4474) overflows fp16 and 4/10 steps get skipped with the scale decaying
  65536 -> 4096. Under autocast the heads come back mixed — `suspicion` and
  `diagnosis_ordinal` stay float32, the rest are float16
- **Batch size 48 at 1280×1024**: peak 76.6 GB of 102 GB, 2.38 s per optimizer
  step (2 micro-batches). `NUM_WORKERS=8` keeps ahead of the GPU (0.29 s/batch
  of dataloading against 1.19 s available); 30 cores are present
- **16-bit survives end to end**: 25,650 distinct levels after
  `preprocess_image`, 11,781 after CLAHE, vs 256 on the old
  `IMREAD_GRAYSCALE` path
- **DICOM conversion invariants** hold across all 8 (scanner × laterality)
  strata: shape, dtype, breast on the correct edge, geometric padding on the
  away-from-breast side, lossless PNG roundtrip

**NOT verified:**

- **any test-set number** — the 12% holdout is deliberately untouched (§2)
- `alpha = POSITIVE_RATE` (the other arm of the §6 experiment)
- whether the `diagnosis` head can be deleted along with its loss (§5)
- TTA, which the paper used over 100 samples and we have never run
- Phase 2 and Phase 3 in any form

## 8. Next steps

Steps 1-6 of the original plan are done: dependencies installed (plus
`scikit-image`, see §6 CLAHE), W&B live, all 20,000 DICOMs converted to 16-bit
PNG, batch size 48 measured, LR 3e-4 established, run 1 complete.

1. **Run the `alpha = POSITIVE_RATE` arm.** One line in `main.ipynb` cell 6;
   the LR and batch size are settled, so this is the cheapest open experiment
   and it is the last thing gating the test set.
2. **Switch the reported diagnosis score to `diagnosis_ordinal`** (§5) — free
   +0.011 AP. Consider `monitor="ordinal_ap"` for checkpoint selection too.
3. **Select checkpoints on a rolling mean of AP, not the raw max.** With a
   0.208-wide CI on 84 positives, single-epoch maxima are mostly noise (§2).
4. **Then, once and only once, evaluate the winner on the test split**, with
   TTA to match the paper's protocol.
5. **Recalibrate the diagnosis probability** before anything product-facing
   consumes it (§6 alpha).
6. Attack the background/GAP dilution (§4): breast-region cropping, or
   attention pooling instead of GAP. This is the standing ceiling on AP.
7. Stratify the patient-level splits by scanner as well as `patient_cancer`
   (§4) — a 6.2x prevalence swing across vendors can drift between train and
   test.
8. GradCAM against the 2,254 radiologist boxes (§4) — pointing-game / IoU. Also
   the cheapest test of whether the model is reading the burned-in corner label.

### Planned experiment sequence (user's roadmap)

1. **Finish the alpha comparison.** Run 1 used `alpha = 1 - POSITIVE_RATE`
   (pos:neg 19.2). Remaining arms worth running:
   - `alpha = POSITIVE_RATE` (0.0495, pos:neg 0.052) — Lin et al.'s α < 0.5
   - `alpha = 0.5` (pos:neg 1.0) — neutral control, isolates γ from α
   ⚠️ A third arm of `alpha=None, weight=1 - POSITIVE_RATE` was planned but
   would be a **silent no-op** — see the `weight` warning in §6 — and is also
   redundant with run 1, whose alpha already equals inverse-frequency.
2. **Pick the winner, then breast-region cropping.** Attacks the 73-85%
   background directly (§4). Note this departs from the paper's "not cropped".
3. **Replace GAP with attention pooling.** The other half of the same problem:
   the median lesion is ~2x2 of ~800 feature cells, so GAP dilutes it ~200x.

Do 2 and 3 as separate runs, not together — they address the same bottleneck by
different means and confounding them wastes the comparison.

### Acceptance gates before "ready for clinical validation"

All baselines below are the **epoch-14 checkpoint on val, scored with
`diagnosis_ordinal`** — model-selection estimates, not test numbers (§2).

**Gate 0 — measurability. This blocks every other gate.** 84 val positives give
an AP CI 0.208 wide, so 0.40 and 0.55 are currently indistinguishable. Width
scales as 1/sqrt(positives):

| positives | AP CI width |
| --- | --- |
| 84 (current val) | 0.208 |
| ~114 (the whole 12% test split) | ~0.178 |
| 988 (every BI-RADS 4+5 image, 5-fold CV) | ~0.061 |

**The test split alone is too small to settle anything** — it holds 600 studies,
~2,400 images, ~114 positives. Publishable numbers need k-fold CV over the full
archive; ~0.06 is the floor this dataset can support.

**Gate 1 — Triage Engine (queue ranking). Nearly met.** Study level, 440
studies, max over the 4 views:

| metric | now | target |
| --- | --- | --- |
| AUROC | 0.854 | >= 0.90 |
| AP | 0.589 | >= 0.65 |
| cancers in top 10% of queue | 54.8% | >= 60% |
| cancers in top 20% | 66.7% | >= 80% |
| cancers in top 50% | 92.9% | >= 98% |
| median queue position of a cancer | 7.8% | <= 10% (FCFS = 50%) |

The number that hurts a triage product is the tail: **7.1% of cancers still sit
in the back half of the queue.** That is the one to drive down, not AP.

**Gate 2 — Diagnostic Co-pilot. This is the binding constraint, ~10x away.**
The auto-draft gate rules out the low-score tail, so it needs high NPV *and*
enough coverage to be worth deploying:

| NPV at the gate | queue auto-handled now | needed |
| --- | --- | --- |
| >= 0.990 (1 miss in 100) | 46.4% | — too risky for cancer |
| >= 0.995 (1 in 200) | **3.2%** | >= 30% |
| >= 0.999 (1 in 1000) | 3.2% | ideal |

Equivalently, specificity at 95% sensitivity is **50.8%** and needs to be ~85%+
to substitute for a read. Ship the Triage Engine first; the Co-pilot is a later
milestone.

**Gate 3 — calibration.** Mean prediction 0.484 against a 0.048 prevalence,
**ECE 0.437**, exactly as §6's alpha analysis predicts. Platt scaling takes it
to 0.018 in-sample. Target ECE <= 0.05 measured on a split *not* used to fit
the calibrator. Blocks both the Abnormality Index and the Co-pilot gate.

**Gate 4 — findings, per class.** Only two classes have usable support: Mass
(AP 0.313, target >= 0.50) and Suspicious Calcification (AP 0.316, target
>= 0.45). The bottom four cannot be validated on this dataset at any split size
— Skin Retraction has 17 positives in all 20,000 images — so they should be
**excluded from clinical claims** rather than reported as failures.

**Gate 5 — not a metric, and not optional.** No AUROC clears these:

- **External validation on Nigerian data** (MAI Lab, §1). VinDr is Vietnamese;
  population and scanner shift is the single biggest unknown for a Lagos/Abuja
  deployment, and nothing in this repo measures it.
- **Subgroup performance** by density, age band, and scanner, with no subgroup
  collapsing. The 6.2x scanner/malignancy confound (§4) must be shown not to be
  the mechanism.
- **GradCAM validated against the 2,254 radiologist boxes** — pointing-game
  >= 70-80%, and demonstrably not firing on the burned-in corner label.
- **A reader study**: radiologist + TULIP versus radiologist alone. That is the
  actual clinical claim, and it is a different experiment from any above.
- **Regulatory pathway** and prospective rather than retrospective evaluation.

These thresholds are engineering targets set to be defensible, not clinical
sign-off criteria — those get set with the clinical partner and the regulator.

### DICOM conversion — decisions and traps

**Done — implemented in `src/dicom.py`, driven by `mammogram_processing.ipynb`.**
Kept here because the reasoning is what makes the code reviewable, and because
re-deriving the vendor quirks is expensive. Measured cost: ~1.0 MB/image,
~0.12 s/image, 21 GB and ~40 min single-process for all 20,000.

Approach: apply the VOI LUT, invert MONOCHROME1, normalise by
`2**BitsStored - 1`, resize preserving aspect with Lanczos, clip, pad.

⚠️ **Resolution: 16:9 is the wrong shape.** Mammograms are portrait ~1.26:1.
Measured padding waste across all 58 shapes:

| target H×W | h/w | padding |
| --- | --- | --- |
| 1080×1920 (16:9 landscape) | 0.562 | **56.2%** |
| 1920×1080 (16:9 portrait) | 1.778 | **27.7%** |
| 1280×720 (16:9 portrait) | 1.778 | **27.7%** |
| 416×320 (paper) | 1.300 | 4.0% |
| **1280×1024 (5:4)** | 1.250 | **2.6%** |
| **1920×1536 (5:4)** | 1.250 | **2.6%** |

28% of every forward pass would convolve black pixels. **Recommendation: store
1920×1536, train 1280×1024** — both 5:4, both real display resolutions, within
0.5% of VinDr's native 1.2564 aspect. Store high and downsample in the Dataset
so the resolution ablation does not require re-converting 20,000 DICOMs.

**Why raise resolution at all** — InceptionResNetV2's effective stride is ~40 px,
so one final-feature-map cell summarises a 40×40 region. Lesion short-side
versus that cell:

| target | median lesion | **below 1 feature cell** |
| --- | --- | --- |
| 416×320 | 24 px | **79.4%** |
| 832×640 | 48 px | 37.8% |
| 1248×960 | 72 px | 18.8% |

At the current resolution the *median* lesion is smaller than one feature cell,
and GAP then averages it across 88 cells. Calcifications are worse: median 19 px,
77% sub-cell.

**DICOM gotchas that silently corrupt data:**

1. `PhotometricInterpretation == "MONOCHROME1"` means inverted — **17% of the
   archive**, all of it Planmed. Left uninverted, roughly one image in six
   enters training as a photographic negative. `PresentationLUTShape ==
   "INVERSE"` agrees with MONOCHROME1 on 68/68 images with no crossover either
   way, so testing `PhotometricInterpretation` alone is sufficient and cannot
   double-invert. Invert *after* the VOI LUT and against the full normalised
   range (`1.0 - x`), never against `array.max()`, which would make the
   transform depend on each image's brightest pixel.
2. Apply the VOI LUT before normalising — **every** image in the archive
   carries one (79% a `VOILUTSequence`, 21% `WindowCenter`/`Width`) and
   `apply_voi_lut` handles both, so it is one call rather than a per-vendor
   branch. Import it from `pydicom.pixels`, not `pydicom.pixel_data_handlers`,
   which is deprecated in pydicom 3.x. Its return dtype is **not stable**:
   uint16 on the LUT-sequence path, float64 on the windowing path.
   `RescaleSlope`/`Intercept` is identity on 120/120 images so no modality LUT
   is needed, but `dicom_to_array` asserts that rather than assuming it.
3. `BitsStored` is usually 12 or 14, not 16 — normalise by `2**BitsStored - 1`,
   not 65535.
4. `pylibjpeg` may be unnecessary — 300/300 sampled files are Explicit VR
   Little Endian, none compressed. Confirm across all 20,000 before dropping
   the dependency.
5. The paper says "mammograms were **not cropped**". Stay uncropped for the
   reproduction; breast-region cropping is a separate later ablation.
6. Padding goes on the side **away from the breast**, decided by laterality.
   Measured over 80 images with zero exceptions: R breasts occupy the right
   edge (left-half intensity fraction 0.011), L breasts the left edge (0.991).
   So padding left for R and right for L means that after `preprocess_image`'s
   existing laterality flip the breast is pinned left with all padding on the
   right and bottom — and `src/utils.py` needed no change, while the stored PNG
   stays faithful to the DICOM's own orientation.
   Beware when checking this: mammograms have 580-1150 columns of genuinely
   black tissue-free background, which swamps the 8-162 px of real geometric
   padding, so verify against the source `Rows`/`Columns` rather than by
   counting all-zero columns.

### Also open

- `main.py` is a placeholder.
- Phase 2 (Classifier) and Phase 3 not started.
- The Triage Engine ranking objective does not exist yet.
- `lr_range_test.csv` and `models/*.pt` are untracked build output; they belong
  in `.gitignore` alongside `wandb/`.
- Evaluation should eventually include ranking metrics — *fraction of cancers
  in the top decile of the queue*, *median queue position of malignant cases vs
  FCFS* — not just diagnosis AP. Those are the numbers that demo the product.

---

## 9. Working style

The user is the ML engineer and reviews all generated code closely; he has
caught real defects in it. He prefers the *why* alongside the change, with
citations to the paper section or a measurement backing the claim.

**Measure, don't reason, when measurement is cheap.** Several conclusions this
project relies on reversed under measurement — including two bugs in generated
helper code (`get_findings_weights` clipping order; the loss-scale cap) that
compiled and ran while being wrong. Prefer a five-line probe over a confident
paragraph, and add asserts that make wrong values announce themselves. The
existing `assert findings_matrix.sum() == 2226` is the model to copy.
