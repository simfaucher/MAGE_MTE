#!/usr/bin/env python3

"""
    Calibration data.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField, ListField, FloatField


class CalibrationData(JsonObject):
    camera_name = StringField()
    K = ObjectListField(list)
    D = ObjectListField(list)
    measure_factor_x = FloatField()
    measure_factor_y = FloatField()
