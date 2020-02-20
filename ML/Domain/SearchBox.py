#!/usr/bin/env python3

"""
    Search box for the scan of an image.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField, EnumStringField
from ML.Domain.Point2D import Point2D
from ML.Domain.RegionOfInterest import RegionOfInterest
from ML.Domain.Step import Step
from ML.Domain.Iteration import Iteration

class SearchBox(JsonObject):
    step = Step()
    iteration = Iteration()
    anchor = Point2D()
    anchor_sight_name = StringField()
