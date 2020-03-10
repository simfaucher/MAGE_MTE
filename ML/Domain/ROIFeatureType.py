#!/usr/bin/env python3

"""
    Features type of ROI.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from enum import Enum, auto

class ROIFeatureType(Enum):
    RAW_PIXELS = "raw_pixels"
    RAW_PIXELS_16 = "raw_pixels_16"
    RAW_PIXELS_32_8 = "raw_pixels_32_8"
    RAW_PIXELS_8_32 = "raw_pixels_8_32"
    COMPOSITE_PROFILE = "composite_profile"
    COLOR_HIST = "color_hist"
    # HOG = "hog"
