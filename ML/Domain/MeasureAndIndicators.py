#!/usr/bin/env python3

"""
    Measurements and indicators.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField, DateTimeField, FloatField
from ML.Domain.RecognitionFlag import RecognitionFlag

class MeasureAndIndicators(JsonObject):
    timespan = FloatField()
    shoke_stroke_measure = FloatField()
    rotation_measure = FloatField()
    light_disturbance_indicator = FloatField()
    vibration_disturbance_indicator = FloatField()
    power_of_recognition_indicator = RecognitionFlag()
