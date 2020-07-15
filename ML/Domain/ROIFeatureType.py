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
    RAW_PIXELS_64_4 = "raw_pixels_64_4"
    RAW_PIXELS_4_64 = "raw_pixels_4_64"
    COMPOSITE_PROFILE = "composite_profile"
    COMPOSITE_PROFILE_64 = "composite_profile_64"
    COMPOSITE_PROFILE_128 = "composite_profile_128"
    COLOR_HIST = "color_hist"
    # HOG = "hog"
