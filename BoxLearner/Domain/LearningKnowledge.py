#!/usr/bin/env python3

"""
    Output of the learning program.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField, DateTimeField
from Domain.ImageSelectionParameters import ImageSelectionParameters
from Domain.Sight import Sight

class LearningKnowledge(JsonObject):
    image_folder = StringField(null=True)
    generation_date = DateTimeField(null=True)
    recognition_selector = ImageSelectionParameters()
    sights = ObjectListField(Sight)
