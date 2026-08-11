# Project Omnia: Entropy-Driven Mixed-Precision Quantization

Entropy-guided mixed-precision post-training quantization (PTQ) for CNNs. Built as a
drop-in replacement for [HAQ](https://github.com/mit-han-lab/haq)'s RL-based
bit-width search: instead of training a DDPG agent (`lib/rl/ddpg.py`,
`lib/env/*.py`) to pick per-layer bit-widths, Project Omnia computes each
layer's activation Shannon entropy `H(X_l)` from a single calibration pass and
assigns bit-widths with a deterministic threshold rule:

```
H(X_l) >= tau  ->  high_bits   (e.g. INT8 weight/activation)
H(X_l) <  tau  ->  low_bits    (e.g. INT4 weight/activation)
```

`tau` is chosen by sweeping a set of candidate values and picking the most
aggressive one (highest fraction of low-bit layers) that still satisfies the
accuracy and bit-width constraints described below.

## Bug fixes applied in this revision

An external code review of the pre-execution codebase (before Section 2 of
`experimental_checklist.md`, "Run Algorithm 1", had ever actually been run)
found four issues worth fixing before running anything for real, plus a few
smaller ones. All are fixed as of this revision:

1. **Entropy calibration crashed on the example commands in this README.**
   `entropy_utils.py`'s `per_tensor` range pass called `torch.quantile()` on
   a layer's entire flattened activation, which exceeds PyTorch's hard
   16,777,216-element `quantile()` limit for realistic layers at the
   default `--calib_batch=25` (e.g. `qmobilenetv2`'s stage-2 expand conv is
   ~30M elements; `qresnet18`/`50`'s stem alone is ~20M). Fixed by
   subsampling before calling `torch.quantile` whenever a tensor is too
   large (`_safe_quantile_1d`/`_safe_quantile_rows` in `entropy_utils.py`).
2. **H(X_l) was computed from the wrong tensor.** `a_bit` quantizes a
   layer's *input* (`QModule._quantize_activation` in `quantize_utils.py`),
   but the entropy hooks read each layer's raw, pre-BatchNorm, pre-
   activation *output*. Fixed by hooking `inp[0]` instead of `out` in both
   `ActivationEntropyCollector` hooks. This also changes what `per_channel`
   mode means -- now per *input* channel -- updated throughout the docs and
   `--entropy_mode` help text.
3. **The calibration set wasn't representative.** It was built as
   `Subset(train_loader.dataset, list(range(calib_size)))`; for
   ImageFolder-backed datasets this pulls entirely from whichever class
   sorts first alphabetically (ImageNet-1k has ~1,300 images/class), using
   train-time-augmented pixels to boot. Fixed by
   `data_utils.get_calibration_loader()`, which samples `--calib_size`
   indices uniformly at random across the whole training split (seeded by
   `--seed`) with the same deterministic preprocessing `val_loader` uses.
4. **Only the first 25 of 100 calibration images ever set a scale/zero-
   point.** `quantize_utils.calibrate()` did `next(iter(loader))` -- one
   batch. Fixed: `calibrate()` now concatenates every batch in the loader
   into a single forward pass, so all of `--calib_size` contributes.

Smaller fixes: `calibrate()` no longer hardcodes `cuda:0` (auto-detects the
model's device); `quantize_utils.py`'s `scikit-learn`/`progress` imports are
now deferred into the (Project-Omnia-unused) k-means functions that need
them, instead of being hard import-time requirements for everyone;
`load_fp32_weights()` now raises instead of silently continuing if a
checkpoint barely matches the constructed model; the `torchvision.models`
registry merge (previously copy-pasted in `entropy_quantize.py`,
`pretrain.py`, and `rl_quantise.py`) is now the single shared
`lib/utils/model_registry.py`; and a couple of vestigial
`torch.autograd.Variable(..., volatile=True)` calls were removed from
`pretrain.py` (harmless no-ops on any current PyTorch, but dead legacy API).

`rl_quantise.py` got the model-registry dedup but is otherwise untouched --
it imports `lib/env/*.py` and `lib/rl/ddpg.py`, which aren't part of this
repo (see below) and so couldn't be verified or fixed.

## What's reused from HAQ vs. new in this repo

| Reused from HAQ (see per-file notes for what changed) | New in Project Omnia |
|---|---|
| `lib/utils/quantize_utils.py` (`QConv2d`, `QLinear` core math unchanged; per-channel weight quantization added; `calibrate()` bug-fixed, see above) | `entropy_quantize.py` (entry point) |
| `lib/utils/data_utils.py` (`get_dataset` -- `imagenet`/`imagenet100`/`imagenet10` branches unchanged; `cifar100`/`imagenet_mini` added) | `lib/utils/entropy_utils.py` (entropy collection, tau sweep) |
| `lib/utils/utils.py` (`Logger`, `AverageMeter`, `accuracy`, unchanged) | `lib/utils/model_registry.py` (shared `torchvision.models` merge helper) |
| `models/mobilenet.py`, `models/mobilenetv2.py`, `models/mobilenetv3.py` (unchanged) | `models/resnet.py`, `models/efficientnet_lite.py` (new architecture families) |
| | `data_utils.get_calibration_loader()` (bug-fixed calibration sampling, see above) |

`lib/rl/ddpg.py` and `lib/env/*.py` are **not** used by `entropy_quantize.py`;
`rl_quantise.py` (HAQ's original RL search) and `pretrain.py` (FP32 training)
are kept in the repo for reference/comparison but are separate pipelines.

## Repo layout

```
entropy_quantize.py          # entry point for entropy-driven PTQ
rl_quantise.py                # original HAQ RL search (unchanged, for comparison)
pretrain.py                   # FP32 training/finetuning (unchanged)
models/
  __init__.py
  mobilenet.py                 # mobilenet / qmobilenet
  mobilenetv2.py                # mobilenetv2 / qmobilenetv2
  mobilenetv3.py                 # mobilenet_v3 (see limitation below)
  resnet.py                       # resnet18 / resnet50 / qresnet18 / qresnet50 (new)
  efficientnet_lite.py             # efficientnet_lite0 / qefficientnet_lite0 (new)
lib/
  utils/
    quantize_utils.py         # QConv2d, QLinear, calibrate (HAQ, +per-channel weights, bug-fixed)
    data_utils.py               # get_dataset (HAQ, +cifar100/imagenet_mini) + get_calibration_loader (new)
    utils.py                      # Logger, AverageMeter, accuracy (HAQ, unchanged)
    entropy_utils.py               # entropy collection + tau sweep (new, bug-fixed)
    model_registry.py               # shared torchvision.models merge helper (new)
```

`lib/__init__.py` and `lib/utils/__init__.py` must exist (can be empty) for
the package imports to resolve.

## Weight/activation quantization scheme

Per experimental_checklist.md Section 1 ("Define the Weight Quantization
Scheme"), the explicit, documented decision (see the header comment of
`lib/utils/quantize_utils.py` for the full rationale) is:

- **Weights**: symmetric, **per-output-channel** by default (`per_channel=True`
  on `QConv2d`/`QLinear`). Pass `per_channel=False` to reproduce HAQ's
  original single-scale-per-tensor behavior, e.g. for the per-channel-vs-
  per-tensor ablation in Section 3.
- **Activations**: symmetric, per-tensor (one scalar `activation_range` per
  layer) -- unchanged from HAQ; per-channel activation quantization was
  deliberately not adopted (see the rationale in `quantize_utils.py`).
- **Bias**: 32-bit, unquantized -- unchanged from HAQ.

## Histogram parameters (entropy estimation)

Per Section 1 ("Set Histogram Parameters"): bin count `B` defaults to **256**
(`--num_bins`, matches an INT8 activation's 256 representable levels). Both
per-tensor and per-channel entropy computation are implemented --
`--entropy_mode per_tensor` (default) or `--entropy_mode per_channel` -- so
the choice can be empirically compared rather than asserted; see the
"Histogram parameters" note at the top of `lib/utils/entropy_utils.py`.

## Requirements

- Python 3, PyTorch + torchvision (CUDA recommended for realistic run
  times; `entropy_quantize.py` also runs on CPU-only setups now that
  `calibrate()` auto-detects the model's device)
- `numpy`
- `scikit-learn` and `progress` are only needed if you actually call
  `quantize_utils.py`'s k-means path (`quantize_model`/
  `kmeans_update_model`) or `pretrain.py`/`rl_quantise.py` (which use
  `progress.bar.Bar` for its own training-loop progress bar) --
  `entropy_quantize.py`'s entropy-driven pipeline doesn't use either, and
  the imports are now deferred so it doesn't require them just to run
- A dataset in one of the formats `get_dataset` understands:
  - **ImageNet-format** (`imagenet`/`imagenet100`/`imagenet10`/`imagenet_mini`):
    `train/` and `val/` subfolders of per-class directories. For
    `imagenet_mini`, download the
    [Kaggle ImageNet-Mini dataset](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
    and point `--dataset_root` at the folder containing `train/`/`val/`.
  - **CIFAR-100** (`cifar100`): downloaded/cached automatically by
    `torchvision.datasets.CIFAR100` under `--dataset_root`; images are
    resized to 224x224 to match the compact vision models' expected input.
- A FP32 checkpoint for the architecture you want to quantize

## Usage

```bash
python entropy_quantize.py \
    --arch qmobilenetv2 \
    --dataset imagenet --dataset_root data/imagenet \
    --resume checkpoints/mobilenetv2/mobilenetv2-150.pth.tar \
    --calib_size 100 --tau_steps 25 \
    --max_acc_drop 0.8 --min_low_bit_frac 0.6 \
    --output save/omnia_mobilenetv2
```

For faster iteration while tuning `tau_steps` or the accuracy/bit-width
constraints, add `--eval_subset 5000` (or similar) so each tau candidate is
evaluated on a subset during the sweep; the final chosen `tau` is always
re-evaluated on the full validation set before the report is written.

### Key arguments

| Argument | Default | Meaning |
|---|---|---|
| `--arch` | `qmobilenetv2` | Model architecture (see supported architectures below) |
| `--resume` | *(required)* | Path to the FP32 checkpoint to quantize |
| `--calib_size` | 100 | Number of images in the calibration stream |
| `--calib_batch` | 25 | Batch size for the calibration stream |
| `--tau_steps` | 25 | Number of tau candidates swept between the observed min/max entropy |
| `--w_bit_low` / `--a_bit_low` | 4 / 4 | Weight/activation bits for layers below tau |
| `--w_bit_high` / `--a_bit_high` | 8 / 8 | Weight/activation bits for layers at/above tau |
| `--max_acc_drop` | 0.8 | Max acceptable Top-1 drop (percentage points) vs. FP32 |
| `--min_low_bit_frac` | 0.6 | Min fraction of layers required at the low-bit setting |
| `--eval_subset` | `None` | If set, subsample validation during the tau sweep for speed |
| `--num_bins` | 256 | Histogram bin count `B` for H(X_l) estimation |
| `--entropy_mode` | `per_tensor` | `per_tensor` or `per_channel` activation entropy computation |

Run `python entropy_quantize.py --help` for the full list.

Other datasets/architectures follow the same pattern, e.g.:

```bash
python entropy_quantize.py \
    --arch qresnet50 \
    --dataset cifar100 --dataset_root data/cifar100 \
    --resume checkpoints/resnet50/resnet50-cifar100.pth.tar \
    --entropy_mode per_channel \
    --output save/omnia_resnet50_cifar100
```

## Supported architectures

Only architectures with a quantized (`QConv2d`/`QLinear`-based) variant work
with `entropy_quantize.py`, since the entropy/bit-assignment pipeline needs
`w_bit`/`a_bit` attributes to write to:

- `qmobilenet` (`models/mobilenet.py`)
- `qmobilenetv2` (`models/mobilenetv2.py`)
- `qresnet18`, `qresnet50` (`models/resnet.py`, new)
- `qefficientnet_lite0` (`models/efficientnet_lite.py`, new)

`resnet18`/`resnet50`/`efficientnet_lite0` (the FP32, `nn.Conv2d`-based
variants of the same architectures) are also registered for use with
`pretrain.py`/`rl_quantise.py`; note these shadow
`torchvision.models.resnet18`/`resnet50` by name (same override mechanism
`mobilenet.py`/`mobilenetv2.py` already rely on) since torchvision's own
`Conv2d` calls are hardcoded and can't be swapped for `QConv2d` without a
local redefinition -- so `pretrained=True` is not implemented for any of
`resnet18`/`resnet50`/`efficientnet_lite0`/their quantized variants; train
an FP32 checkpoint locally with `pretrain.py` first.

**`mobilenetv3.py` does not currently have a quantized variant** — its
`MobileBottleneck` block hardcodes `conv_layer = nn.Conv2d` and never imports
`QConv2d`/`QLinear`, so there is no `qmobilenet_v3` entry point. `--arch
qmobilenetv3` will not resolve. Adding one would mean threading a
`conv_layer` argument through `MobileNetV3`/`MobileBottleneck` the way
`mobilenet.py`/`mobilenetv2.py` already do.

## Output

Each run writes to `<output>/`:

- `omnia_report.json` — FP32/quantized Top-1, accuracy drop, chosen tau,
  per-layer entropy and bit-width strategy, compression estimate, and
  PASS/FAIL against the `--max_acc_drop` / `--min_low_bit_frac` values used
  for that run (plus the fixed 2.2x compression / 180s calibration-time
  targets from the project brief).
- `omnia_quantized.pth.tar` — `state_dict`, chosen `strategy`, and `tau`.
- `log.txt` — per-tau sweep log (tau, acc, acc_drop, low_bit_frac, feasible).

## Known limitations

- **Compression numbers are a bit-width estimate, not an on-disk size
  reduction.** `QConv2d`/`QLinear` fake-quantize on the fly (straight-through
  estimator) and always store full FP32-shaped weight tensors — the
  `compression_vs_int8`/`compression_vs_fp32` figures in the report describe
  the theoretical footprint *if* the chosen bit-widths were packed, not the
  actual size of `omnia_quantized.pth.tar`. Real latency/memory numbers
  require wiring in `lib/simulator/lookup_tables` (`HAS_HW_SIMULATOR`); the
  weight-size estimate is a fallback.
- The tau sweep re-calibrates and re-evaluates the model once per candidate
  (`--tau_steps`, default 25); use `--eval_subset` to keep this affordable
  on the full ImageNet validation set. `calibrate()` now runs the *entire*
  calibration set as a single forward pass each time (see "Bug fixes"
  above), so this scales with `--calib_size` as well as `--tau_steps` --
  keep `--calib_size` at "tiny calibration subset" scale (the brief's
  target is 100) rather than pushing it very high.
- **Per-channel weight quantization is now the default** (see "Weight/
  activation quantization scheme" above), which is a behavior change
  relative to upstream HAQ's per-tensor scheme -- if you're diffing
  against older HAQ numbers/checkpoints, either regenerate them or pass
  `per_channel=False` when constructing `QConv2d`/`QLinear` to match the
  original per-tensor behavior.
- `cifar100`/`imagenet_mini` require either network access (CIFAR-100's
  `download=True`) or a manually-downloaded dataset (ImageNet-Mini via
  Kaggle) at `--dataset_root`; neither is bundled with this repo.
