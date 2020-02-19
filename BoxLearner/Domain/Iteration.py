#!/usr/bin/env python3

"""
    Defines a search box step.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField, EnumStringField

class Iteration(JsonObject):    
    _x = IntegerField(serialized_name="x")
    _y = IntegerField(serialized_name="y")

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if value >= 0:
            self._x = value
        else:
            self._x = 0

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if value >= 0:
            self._y = value
        else:
            self._y = 0
