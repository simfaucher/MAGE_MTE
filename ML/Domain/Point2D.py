#!/usr/bin/env python3

"""
    Defines a 2D point.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField, EnumStringField

class Point2D(JsonObject):    
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
    
    def __add__(self, other):
        point = Point2D()
        if isinstance(other, Point2D):
            point.x = self.x + other.x
            point.y = self.y + other.y
        
        elif isinstance(other, tuple):
            point.x = self.x + other[0]
            point.y = self.y + other[1]
        
        return point