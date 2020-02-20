#!/usr/bin/env python3

"""
    Defines a region of interest.

    Project Nose Landing Gear Video Measurement for ATR
    Created on Mon Oct 21 2019 by Frank Ben Zaquin, Fabien Monniot
    Copyright (c) 2019 Altran Technologies
"""

from pykson import JsonObject, IntegerField, StringField, ObjectListField, EnumStringField, FloatField
from ML.Domain.ImageFilterType import ImageFilterType
from ML.Domain.Image import Image
from ML.Domain.ROIFeatureType import ROIFeatureType

class RegionOfInterest(JsonObject):
    name = StringField()
    tolerance = FloatField()
    image_filter_type = EnumStringField(ImageFilterType, serialized_name="filter")
    images = ObjectListField(Image, null=True)
    feature_type = EnumStringField(ROIFeatureType, serialized_name="feature_type")

    _x = IntegerField(serialized_name="x")
    _y = IntegerField(serialized_name="y")
    _width = IntegerField(serialized_name="width")
    _height = IntegerField(serialized_name="height")

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if value >= 0:
            self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if value >= 0:
            self._y = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value > 0:
            self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value > 0:
            self._height = value
