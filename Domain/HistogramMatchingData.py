from pykson import JsonObject, ListField

class HistogramMatchingData(JsonObject):
    values = ListField(int)
    counts = ListField(int)
    quantiles = ListField(float)
