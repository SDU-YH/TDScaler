from collections import deque

class RunningAverage:
    def __init__(self, size):
        self.size = size
        self.q = deque()
        self.sum = 0

    def add(self, val):
        self.q.append(val)
        self.sum += val
        if len(self.q) > self.size:
            self.sum -= self.q.popleft()

    def mean(self):
                           
        return self.sum / max(len(self.q), 1)
