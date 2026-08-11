# Experimental & Manuscript Checklist
### Information-Entropy Driven Adaptive Mixed-Precision Quantization

---

## 1. Finalize the Experimental Setup (Pre-Computation)

- [x] **Select Architectures** — `models/resnet.py` (`resnet18`/`resnet50`/`qresnet18`/`qresnet50`) and `models/efficientnet_lite.py` (`efficientnet_lite0`/`qefficientnet_lite0`) added alongside the existing MobileNet/MobileNetV2, giving three quantizable architecture families. (MobileNetV3 remains FP32-only -- see README "Supported architectures".)
- [x] **Prepare the Standard Benchmarks** — `lib/utils/data_utils.py`'s `get_dataset()` now has `cifar100` and `imagenet_mini` branches.
  - **CIFAR-100**: `torchvision.datasets.CIFAR100`, resized to 224×224, CIFAR-100-specific normalization. Needs network access (or a pre-populated `--dataset_root`) to actually download — not run in this pass.
  - **ImageNet-Mini**: `ImageFolder` over `--dataset_root/{train,val}`, same transforms as the `imagenet` branch. You still need to download it from Kaggle yourself; this environment has no network access to do that for you.
- [x] **Define the Weight Quantization Scheme** — documented (and implemented) in `lib/utils/quantize_utils.py`: symmetric weights & activations; weights per-output-channel by default (`per_channel=True`, with a `per_channel=False` fallback for the Section 3 ablation); activations stay per-tensor. Full rationale in that file's header comment and in README's "Weight/activation quantization scheme" section.
- [x] **Set Histogram Parameters** — `B=256` default kept (`--num_bins`); per-tensor vs. per-channel entropy computation is now a runtime choice (`--entropy_mode`) in `lib/utils/entropy_utils.py` rather than a single hardcoded behavior, so the two can be run back-to-back and diffed. Actually *empirically justifying* the choice (running both modes on real calibration data and comparing the resulting tau/accuracy) still needs a GPU + dataset and hasn't been executed here.

## 2. Execute the Calibration & Thresholding Loop

- [ ] **Run Algorithm 1** — execute the accuracy-constrained threshold selection across chosen models with a ~100-image calibration set.
- [ ] **Track Iterations** — record the maximum iterations ($N_{max}$) required for convergence.
- [ ] **Verify Timing** — confirm total wall-clock calibration time stays strictly under the 3-minute target.

## 3. Run Baseline and Ablation Experiments

- [ ] **Benchmark Baselines** — evaluate against uniform INT8, uniform INT4, HAQ, HAWQ, TensorRT-style KL-divergence, and at least one existing entropy-based method, under identical conditions.
- [ ] **Conduct Ablations** — entropy-based assignment vs. weight-magnitude/range-based vs. random assignment, holding INT4 fraction fixed.
- [ ] **Test Thresholds** — per-layer-group threshold vs. global threshold.
- [ ] **Check Sensitivity** — sweep calibration set size across 50, 100, 200, 500 images to verify entropy estimator stability.
- [ ] **Ensure Statistical Rigor** — run all evaluations across ≥3 random seeds/calibration-set draws; report mean ± std for accuracy degradation.

## 4. Gather Hardware Latency Metrics

- [ ] **Measure Latency** — obtain latency/throughput numbers; explicitly state whether they come from measured silicon on a named board or a cited, cycle-accurate simulator.
- [ ] **Detail Custom Datapaths** — if using standard Vitis-AI DPU IP (no out-of-the-box packed INT4 support), document the custom HLS datapath used.
- [ ] **Label Accurately** — until real hardware metrics are finalized, report all latency figures strictly as "modeled cycle counts," not "measured latency."

## 5. Manuscript Finalization (Post-Results)

- [ ] **Populate Section V** — fill results tables/figures: accuracy retention, INT4 coverage, memory reduction, latency speedup, domain-shift degradation.
- [ ] **Create Visuals** — qualitative figure of per-layer entropy vs. assigned bit-width for ≥1 architecture.
- [ ] **Update the Abstract** — insert 2–3 sentences summarizing final quantitative results (top-1 accuracy retention, memory reduction, latency speedup).
- [ ] **Refine the Introduction** — explicitly state which design choices differentiate this work from [5] (e.g., pre-quantization activation entropy vs. their formulation).
- [ ] **Complete Citations** — confirm exact author lists for [5] and [6]; add the full citation for the URPC2021 challenge report.
