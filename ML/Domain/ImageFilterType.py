#!/usr/bin/env python3

"""
    Type of the image filter.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from enum import Enum, auto

class ImageFilterType(Enum):
    GREY_SCALE = "grey_scale"
    KIRSCH = "kirsch"
    CANNY = "canny"
    LSD = "lsd"
    EMBOSS = "emboss"
    BINARISATION = "binarisation"
    COLOR = "color"
    HSV = "hsv"
    POSTERISATION = "posterisation"
