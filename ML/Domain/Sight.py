#!/usr/bin/env python3

"""
    A sight contains a anchor and a region of interest.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField, EnumStringField
from ML.Domain.Point2D import Point2D
from ML.Domain.RegionOfInterest import RegionOfInterest
from ML.Domain.SearchBox import SearchBox

class Sight(JsonObject):
    name = StringField()
    width = IntegerField()
    height = IntegerField()
    anchor = Point2D()
    roi = ObjectListField(RegionOfInterest)
    search_box = SearchBox()
