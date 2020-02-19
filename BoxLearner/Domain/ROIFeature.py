#!/usr/bin/env python3

"""
    Feature outputed from a ROI.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ListField, EnumStringField, FloatField, ObjectListField
from Domain.ROIFeatureType import ROIFeatureType

class ROIFeature(JsonObject):
    feature_type = EnumStringField(ROIFeatureType, serialized_name="type")
    feature_vector = ListField(float)