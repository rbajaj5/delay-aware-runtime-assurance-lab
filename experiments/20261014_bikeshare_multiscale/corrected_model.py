"""Versioned correction; the retained 992-episode implementation is untouched."""
from routing import Router


class CorrectedRouter(Router):
    def itinerary(self, path, departure, speed, crossing, start, end):
        # None denotes no closure, not an ordinary street whose crossing is None.
        if crossing is None:
            start = end = 0
        return super().itinerary(path, departure, speed, crossing, start, end)
