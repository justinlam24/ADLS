# ADLS
## Lab 1 - Modelling Compression via Quantisation and Pruning
### Objectives
 - The primary aim of the lab is to explore the model comression techniques using Mase.
 - These aim to reduce the computuational and storage footprint of a BERT model, while also mainitaining accuracy.
 - The specicic goals are:
    - Quantisation exploration
        - Implement Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) to transform the model from floating-point to fixed-point precision (4-bit to 32-bit).
    - Performance recovery
        -  Evaluate the effectiveness of QAT in recovering accuracy drops caused by lower bit-width precision.
    - Unstructured pruning
        - Apply unstruyctured pruning to a previously quantised model
    - Comparitive analysis
        - Evaluate different pruning methods, like random and L1-norm.


### Backgroudn Information
#### Model Quantisation
 - This is the process of mapping high-precision floating point numbers, like float32, to lower precision formats, including fixed-point integers.
 - There are two methods:
    - QAT (Quantisation Awareness Training)
        - Fine-tuned while simulation quantusating effects in the forward pass
        - Allows weights to adapt to rounding errors.
        - Often recovering most, if not all, original precision performance. 
    - PTQ (Post-Training Quantisation)
        - Quantised when training is complete.
        - Fast
        - Significant accuracy loss
#### Unstryctyred Pruning
 - Model complexity gets reduced by setting specifuc weights to zero.
    - Sparcity: The proportion of weights removed
    - L1-norm pruning: A magntiude-based approach where weights with the smallest absolute values are removed.
    - Random pruning: Weights are removed regardless of their value, serving as a baseline to evaluate more intelligent pruning strategies.

#### The MASE Framework
 - Mase facilitates these transformations using MaseGraph.
 - This allows for passes, like `quantize_transform_pass` and `prune_transform_pass`.
 - These passes modify the internal operators to support fixed-point arithmetic or weight masking while remining compatible with standard training loops.
    - An example of this is the HuggingFace `Trainer`

### Lab 1 Implementation Task 1

#### Summary of Results:

    Width      PTQ Accuracy    QAT Accuracy    QAT Gain       

    4          0.5000          0.5000          0.0000         
    8          0.7939          0.8405          0.0466         
    12         0.8317          0.8414          0.0097         
    16         0.8344          0.8419          0.0075         
    20         0.8351          0.8417          0.0066         
    24         0.8350          0.8417          0.0068         
    28         0.8351          0.8418          0.0066         
    32         0.8349          0.8418          0.0069         

#### Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT) Accuracy Trends:

* QAT consistently outperforms PTQ at each width, but the gap is most pronounced at lower bits (4, 8 bits).
* Both methods converge to similar accuracy as the widths increases (16 bits onwards), indicating quantization error becomes negligable at higher precisions.

#### Precision and Trade Offs:

* Both PTQ and QAT plateau around 12 bits: increasing bit-width beyond this point doesn't yield significant accuracy gains.
* At higher bit-width, the quantized representation is close to full precision, so both methods approached original model accuracy.

    Lower Bit-Widths:
    * Reduce model size and memory bandwidth
    * Faster inference
    * Higher quantization error and accuracy loss (especially PTQ)

    Higher Bit-Widths:
    * Approach full-precision accuracy
    * Speedup decreases
    * Compression ratio drops

#### Quantization:
* Lowering width increases quantization errors (rounding, clipping etc)
* Fractional widths control dynamic range and granularity of quantized values.
    * Too small: loses range
    * Too large: loses precision

#### Conclusion:

QAT is essential for aggresive quantization, while PTQ is sufficient at higher bit-widths. Both methods benefit from careful selection of quantization parameters and mixed-precisions to further optimize the trade-offs between accuracy and efficiency.

### Lab 1 Implementation Task 2

#### Summary of Results:

    RANDOM PRUNING:
    Sparsity 0.1: 0.8384
    Sparsity 0.3: 0.8060
    Sparsity 0.5: 0.7309
    Sparsity 0.7: 0.5030
    Sparsity 0.9: 0.4992

    L1-NORM PRUNING:
    Sparsity 0.1: 0.8573
    Sparsity 0.3: 0.8508
    Sparsity 0.5: 0.8332
    Sparsity 0.7: 0.8012
    Sparsity 0.9: 0.5539

#### Random vs L1-Norm Pruning Accuracy:

* Random Pruning: random removal of weights, could eliminate weights that are critical for the model's predictions, leading to rapid drop in accuracy as sparsity increases.
* L1-Norm Pruning: removes weights starting from the smallest absoulute values, which are typically less important for the model output. This preserves the most signicant weights to maintain higher accuracy at the same sparsity level.

#### Effect of increasing Sparsity:
* Sparsity increases -> more weights are pruned, accuracy decreases.
* At lower sparsity the model still tolerates the removal of unimportant weights, however at higher sparsity as the important weights are pruned the accuracy has a steep decline.
* L1-Norm Pruning has a distinct advantage against Random Pruning, as seen in the plot even at high sparsity it could still preserve some structure.

#### Accuracy and Trade Offs:

* Higher Sparsity:
    * Smaller, faster models
    * Greater risk of accuracy loss
* Lower Sparsity:
    * Preserves accuracy
    * Yields less compression

#### Conclusion:
L1-Norm Pruning is more effective than Random Pruning, especially at high sparsities since it preserves the most important weights. As sparsity indreases, accuracy drops nonlinearly, with Random Pruning demonstrating degradation in accuracy to chance level (0.5).

## Lab 2 - Neural Architecture Search (NAS)

### Objectives
* The primary aim of this lab is to learn how to use Mase to conduct a Neural Architecture Search (NAS) on a BERT model.
* These techniques aim to automate the discovery of optimal model configurations for specific hardware or performance constraints.
* The specific goals are:
    * **Integration with Optimization Frameworks**: 
        * Understand how to integrate Mase with Optuna to search for optimized Bert models.
    * **Sampler Comparison**: 
        * Explore and compare different Optuna samplers, specifically `GridSampler` and `TPESampler`, against the baseline `RandomSampler`.
    * **Compression-Aware Search**: 
        * Implement a search flow where quantization and pruning are integrated directly into the objective function to identify architectures that are inherently robust to compression.
    * **Performance Evaluation**: 
        * Compare the accuracy of standard NAS models against compression-aware models, with and without post-compression fine-tuning.

---

### Background Information

#### Neural Architecture Search (NAS)
* NAS is a technique for automating the design of neural networks to find the best configuration of hyperparameters and layer choices.
* **Search Space**: A defined dictionary of possible values for parameters such as hidden size, number of layers, number of attention heads, and intermediate sizes.
* **Objective Function**: A function that constructs a model from the search space, trains it, and returns a metric (e.g., accuracy) for Optuna to maximize.

#### Optuna Samplers
* Optuna provides different algorithms to navigate the search space:
    * **GridSampler**: Iterates through every possible combination of hyperparameters.
    * **RandomSampler**: Chooses random combinations in each iteration as a baseline.
    * **TPESampler (Tree-structured Parzen Estimator)**: Uses a Bayesian optimization algorithm to choose hyperparameter values based on the results of previous trials.

#### Compression-Aware Search
* Traditionally, models are searched for first and compressed later. However, different architectures have varying sensitivities to compression.
* **Compression-Aware Search** runs the `CompressionPipeline` (quantization and pruning) during each trial of the NAS.
* This ensures the search finds an architecture that maintains high accuracy specifically *after* it has been quantized and pruned.

#### The MASE Integration
* Mase provides a `CompressionPipeline` that can be called within the Optuna objective function.
* This allows for the evaluation of a model's final "compressed" accuracy immediately after it has been sampled and trained.
* It supports exporting the best-found architecture as a checkpoint for future deployment.

![Task 1 cumulative max accuracy](https://raw.githubusercontent.com/justinlam24/ADLS/main/figures/Lab2_Fig1.png)



---
## Lab 3: Mixed Precision Search

### Objectives
* **Core Aim**: Use the **Mase** framework to discover optimal **mixed precision** quantization schemes.
* **Optimization Goal**: Balance model accuracy and hardware efficiency by assigning higher precision only to sensitive network components.
* **Key Tasks**:
    * **Granular Exploration**: Allow layer-specific variation in total bit-widths and fractional widths.
    * **Format Diversity**: Expand search beyond integers to include **Minifloat**, **BlockFP**, **BlockLog**, and **Binary**.
    * **Automated NAS**: Use **Optuna** to automate efficient bit-width allocation across all layers.
    * **Analysis**: Evaluate and plot how precision formats impact accuracy throughout trial iterations.

---

### Background Information

#### Mixed Precision Quantization
* **Definition**: Assigns different numerical bit-widths to individual layers or tensors rather than a uniform global setting.
* **Sensitivity**: Early layers (raw data processing) are often highly sensitive to noise, whereas deeper layers may tolerate more aggressive compression.
* **Benefits**: 
    * **Efficiency**: Reduces computational and memory load.
    * **Accuracy**: Maintains performance by focusing bits where they matter most.

#### Supported Precision Formats in Mase
* **LinearInteger**: Standard fixed-point with configurable fractional widths.
* **Minifloat (IEEE/Denorm)**: Low-bit floating-point for diverse dynamic ranges.
* **Block Floating Point (BlockFP)**: A block of numbers sharing a single exponent to save space.
* **Binary/Ternary**: Ultra-low 1 or 2-bit quantization for maximum storage reduction.

#### Optimization with Optuna
* **Trial-based Search**: Executes multiple trials using samplers like **TPESampler** to test suggested precision configurations.
* **Objective Function**: Constructs a model, trains briefly on IMDb data, and returns accuracy to guide subsequent trials.
* **Sensitivity Analysis**: The process automatically identifies layer importance; critical layers receive higher bit-widths to satisfy accuracy constraints.

#### Mase Implementation
* **Define-by-Run**: Models and their quantization schemes are constructed dynamically during search.
* **Automation**: Uses `trial.suggest_categorical` to iterate through layer types and precision parameters for every `nn.Linear` module.

---


### Lab 3 - Tutorial 6: Mixed-Precision Search (Tasks 1–2)

This note summarizes the two implementation extensions requested on top of **Tutorial 6** (mixed-precision search for `Linear` layers in Mase), and reports the resulting Optuna search behavior using the generated plots.

---

### Task 1 — Per-layer Integer precision (width & fractional width)

#### 1a) Change requested

In the original Tutorial 6 setup, all layers mapped to `LinearInteger` share the *same* width and fractional-width. This is suboptimal because different `Linear` layers can have __different quantization sensitivity__.

**Modification**: For every `torch.nn.Linear` that is replaced with `LinearInteger`, sample **per-layer**:

- width ∈ **{8, 16, 32}**
- fractional width ∈ **{2, 4, 8}**

These are exposed as **additional Optuna hyperparameters** *per layer*, so the sampler can choose independently per module.

#### 1b) Result plot: cumulative max accuracy

The figure below shows the **cumulative best** evaluation accuracy during the Optuna run:

- x-axis: trial index  
- y-axis: best accuracy observed up to that trial

![Task 1 cumulative max accuracy](https://raw.githubusercontent.com/justinlam24/ADLS/main/figures/lab3_fig1.png)

**Interpretation**: The curve is a staircase (typical for “best-so-far” plots). Improvements occur when a trial discovers a better per-layer width / fractional-width allocation.

**Why the curve looks like this**: This is a cumulative-best (“best so far”) plot, so it increases only when a trial finds a better configuration. Most trials reuse similarly good per-layer fixed-point settings, producing plateaus, and occasional jumps occur when the sampler hits a better match between a layer’s numeric range and its chosen `{width, frac_width}` (reducing saturation/rounding error in the most sensitive layers).


---

### Task 2 — Extend search to all supported Linear precisions

#### 2a) Change requested

Tutorial 6 imports multiple precision variants, but only searches between:

- full-precision `torch.nn.Linear`
- `LinearInteger`

**Modification**: Extend the search to include all supported Mase linear precisions, e.g.

- `LinearMinifloatDenorm`, `LinearMinifloatIEEE`
- `LinearBlockFP`, `LinearBlockLog`
- `LinearLog`
- `LinearBinary`, `LinearBinaryScaling`, `LinearBinaryResidualSign`

This required updating the model constructor to pass the **expected `config` fields** for each precision type (since each module family has different required config keys).

#### 2b) Result plot: cumulative max accuracy per precision

The following plot compares precision families by running **separate Optuna studies** per precision type and plotting:

- x-axis: number of trials (per precision)  
- y-axis: best accuracy observed so far (per precision)

![Task 2 cumulative max accuracy per precision](https://raw.githubusercontent.com/justinlam24/ADLS/main/figures/lab3_fig2.png)

**Interpretation**:

- The plot provides a direct comparison of which precision family reaches higher accuracy within the same trial budget.
- In this run, the best-performing group clusters near the top curve band, while some precision types lag (notably the pure __binary and logarithmic__ variant in this configuration).

---