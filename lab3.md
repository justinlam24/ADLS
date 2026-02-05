# Lab 3 - Tutorial 6: Mixed-Precision Search (Tasks 1–2)

This note summarizes the two implementation extensions requested on top of **Tutorial 6** (mixed-precision search for `Linear` layers in Mase), and reports the resulting Optuna search behavior using the generated plots.

---

## Task 1 — Per-layer Integer precision (width & fractional width)

### 1a) Change requested

In the original Tutorial 6 setup, all layers mapped to `LinearInteger` share the *same* width and fractional-width. This is suboptimal because different `Linear` layers can have __different quantization sensitivity__.

**Modification**: For every `torch.nn.Linear` that is replaced with `LinearInteger`, sample **per-layer**:

- width ∈ **{8, 16, 32}**
- fractional width ∈ **{2, 4, 8}**

These are exposed as **additional Optuna hyperparameters** *per layer*, so the sampler can choose independently per module.

### 1b) Result plot: cumulative max accuracy

The figure below shows the **cumulative best** evaluation accuracy during the Optuna run:

- x-axis: trial index  
- y-axis: best accuracy observed up to that trial

![Task 1 cumulative max accuracy](https://github.com/justinlam24/ADLS/blob/main/figures/lab3_fig1.png)

**Interpretation**: The curve is a staircase (typical for “best-so-far” plots). Improvements occur when a trial discovers a better per-layer width / fractional-width allocation.

**Why the curve looks like this**: This is a cumulative-best (“best so far”) plot, so it increases only when a trial finds a better configuration. Most trials reuse similarly good per-layer fixed-point settings, producing plateaus, and occasional jumps occur when the sampler hits a better match between a layer’s numeric range and its chosen `{width, frac_width}` (reducing saturation/rounding error in the most sensitive layers).


---

## Task 2 — Extend search to all supported Linear precisions

### 2a) Change requested

Tutorial 6 imports multiple precision variants, but only searches between:

- full-precision `torch.nn.Linear`
- `LinearInteger`

**Modification**: Extend the search to include all supported Mase linear precisions, e.g.

- `LinearMinifloatDenorm`, `LinearMinifloatIEEE`
- `LinearBlockFP`, `LinearBlockLog`
- `LinearLog`
- `LinearBinary`, `LinearBinaryScaling`, `LinearBinaryResidualSign`

This required updating the model constructor to pass the **expected `config` fields** for each precision type (since each module family has different required config keys).

### 2b) Result plot: cumulative max accuracy per precision

The following plot compares precision families by running **separate Optuna studies** per precision type and plotting:

- x-axis: number of trials (per precision)  
- y-axis: best accuracy observed so far (per precision)

![Task 2 cumulative max accuracy per precision](https://github.com/justinlam24/ADLS/blob/main/figures/lab3_fig2.png)

**Interpretation**:

- The plot provides a direct comparison of which precision family reaches higher accuracy within the same trial budget.
- In this run, the best-performing group clusters near the top curve band, while some precision types lag (notably the pure __binary and logarithmic__ variant in this configuration).

---

## Artifacts

- Task 1 plot: `task1_cummax_accuracy.png`
- Task 2 plot: `task2_cummax_accuracy_per_precision.png`
