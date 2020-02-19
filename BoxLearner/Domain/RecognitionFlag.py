#!/usr/bin/env python3

"""
    Flags of recognition

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from enum import Enum, auto

class RecognitionFlag(Enum):
    GREEN = "green"
    LIGHTGREEN = "lightgreen"
    ORANGE = "orange"
    RED = "red"
