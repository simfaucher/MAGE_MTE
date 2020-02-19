#!/usr/bin/env python3

"""
    Image class, category.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField
from Domain.ROIFeature import ROIFeature
from Domain.Point2D import Point2D

class ImageClass(JsonObject):
    id = IntegerField()
    name = StringField()
