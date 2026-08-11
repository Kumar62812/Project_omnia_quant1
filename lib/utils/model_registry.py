# lib/utils/model_registry.py
#
# Project Omnia addendum (integration fix, added during code review).
#
# The ~8-line block that merges torchvision.models with this repo's local
# `models` package -- so `models.__dict__['qmobilenetv2']` resolves, and
# local resnet18/resnet50 shadow torchvision's own resnet18/resnet50 by
# name (see README.md's "Supported architectures" section) -- was
# previously copy-pasted verbatim into entropy_quantize.py, pretrain.py,
# and rl_quantise.py. Centralized here so there's exactly one place to
# look at (or change) it. Behavior is unchanged from the original: this
# still mutates torchvision.models' own namespace, the same pattern
# mobilenet.py/mobilenetv2.py already relied on -- just no longer
# duplicated three times.

import torchvision.models as tv_models


def build_model_registry(customized_models):
    """
    customized_models: the repo's local `models` package, i.e. whatever
    you get from `import models as customized_models` at the top of an
    entry-point script.

    Copies every lowercase, callable name from `customized_models` into
    torchvision.models' own namespace, overwriting any torchvision name
    of the same name. This is how `resnet18`/`resnet50` end up resolving
    to this repo's from-scratch, QConv2d-swappable definitions
    (models/resnet.py) instead of torchvision's own nn.Conv2d-hardcoded
    ones -- see README.md's "Supported architectures" section for why
    torchvision's own resnet18/50 can't be reused directly.

    Returns (model_names, models):
      - model_names: combined, sorted list of both torchvision's and the
        local package's model names, suitable for argparse's `choices=`.
      - models: the (now patched) torchvision.models module, so callers
        can keep doing `models.__dict__[args.arch](...)` exactly as
        before.
    """
    default_model_names = sorted(
        name for name in tv_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(tv_models.__dict__[name]))
    customized_model_names = sorted(
        name for name in customized_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(customized_models.__dict__[name]))
    for name in customized_models.__dict__:
        if (name.islower() and not name.startswith("__")
                and callable(customized_models.__dict__[name])):
            tv_models.__dict__[name] = customized_models.__dict__[name]
    model_names = default_model_names + customized_model_names
    return model_names, tv_models
