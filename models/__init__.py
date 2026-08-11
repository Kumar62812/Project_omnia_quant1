# import customized models here
from .mobilenet import *
from .mobilenetv2 import *
from .mobilenetv3 import *
# Project Omnia addendum (experimental_checklist.md Section 1: "Select
# Architectures") -- ResNet-18/50 and EfficientNet-Lite0, each with a
# quantizable q* variant, so generalization claims span more than the
# single MobileNet family.
from .resnet import *
from .efficientnet_lite import *
