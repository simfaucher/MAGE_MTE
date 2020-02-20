#!/usr/bin/env python3

"""
    Output data from the BoxLearner scan method.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ListField, DateTimeField, FloatField, BooleanField, EnumStringField
from ML.Domain.RecognitionFlag import RecognitionFlag
from ML.Domain.Point2D import Point2D

class LearnerMatch(JsonObject):
    image_name = StringField()
    success = BooleanField()
    anchor = Point2D()
    hit_in = IntegerField()
    sum_distances = FloatField()
    max_distance = FloatField()
    power_of_recognition = EnumStringField(RecognitionFlag)
    error = FloatField()
    error_x = FloatField()
    error_y = FloatField()
    roi_distances = ListField(float)
    roi_classes = ListField(int)
    predicted_class = IntegerField()
    reduced = BooleanField()
