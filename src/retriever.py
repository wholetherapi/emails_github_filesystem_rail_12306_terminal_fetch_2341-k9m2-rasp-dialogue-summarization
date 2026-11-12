"""Dialogue turn retrieval for sentence planning."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Turn:
    speaker: str
    text: str


class TurnIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors: List[np.ndarray] = []
        self.turns: List[Turn] = []

    def add_dialogue(self, turns: List[Turn]):
        for turn in turns:
            self.vectors.append(self.embedder.encode(turn.text))
            self.turns.append(turn)

    def search(self, query: str, k: int = 8):
        if not self.vectors:
            return []
        query_vector = self.embedder.encode(query)
        stacked = np.stack(self.vectors)
        scores = stacked @ query_vector
        order = np.argsort(-scores)[:k]
        return [self.turns[i] for i in order]
