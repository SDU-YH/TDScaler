from abc import ABC, abstractmethod

class EpsilonAnneal(ABC):
    @abstractmethod
    def anneal(self):
        pass

class Constant(EpsilonAnneal):
    def __init__(self, start):
        self.val = start

    def anneal(self):
        pass

class LinearAnneal(EpsilonAnneal):
    pass

    def __init__(self, start: float, end: float, duration: int):
        self.val = start
        self.min = end
        self.duration = duration

    def anneal(self):
        self.val = max(self.min, self.val - (self.val - self.min) / self.duration)
