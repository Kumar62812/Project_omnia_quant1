"""Explicit local architecture registry for Project Omnia.

The previous implementation mutated torchvision.models globally. Project Omnia
now returns a plain mapping so importing the research code has no global side
effects.
"""


def build_model_registry(customized_models):
    registry = {}
    for name, value in vars(customized_models).items():
        if name.startswith('_') or not callable(value):
            continue
        registry[name] = value
    return sorted(registry), registry
