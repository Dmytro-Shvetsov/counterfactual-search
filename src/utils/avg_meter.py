from collections import defaultdict


class AvgMeter:
    def __init__(self) -> None:
        self.reset()

    def update(self, new_data):
        for k, v in new_data.items():
            self.data[k] += v
        self.steps += 1
    
    def average(self):
        return {k: v / self.steps for k, v in self.data.items()}
    
    def reset(self):
        self.data = defaultdict(int)
        self.steps = 0
