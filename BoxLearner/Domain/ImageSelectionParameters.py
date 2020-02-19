#!/usr/bin/env python3

"""
    Settings of the recognition program.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField, EnumStringField, FloatField
from Domain.ROIFeatureType import ROIFeatureType

class ImageSelectionParameters(JsonObject):
    uncertainty = FloatField()
    feature_type = EnumStringField(ROIFeatureType, serialized_name="type")