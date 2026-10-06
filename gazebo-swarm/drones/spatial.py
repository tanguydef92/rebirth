"""Broad phase for continuous, simultaneous drone movements (stdlib only)."""
from collections import defaultdict
import itertools
import math


class SegmentIndex:
    def __init__(self, separation):
        self.margin = separation / 2
        self.width = separation * 2
        self.buckets = defaultdict(set)
        self.keys = {}

    def cells(self, start, end):
        return itertools.product(*(range(math.floor((min(start[k], end[k]) - self.margin) / self.width),
                                         math.floor((max(start[k], end[k]) + self.margin) / self.width) + 1)
                                   for k in range(3)))

    def update(self, number, start, end):
        for key in self.keys.get(number, ()):
            self.buckets[key].discard(number)
            if not self.buckets[key]:
                del self.buckets[key]
        keys = tuple(self.cells(start, end))
        self.keys[number] = keys
        for key in keys:
            self.buckets[key].add(number)

    def query(self, start, end):
        result = set()
        for key in self.cells(start, end):
            result.update(self.buckets.get(key, ()))
        return result
