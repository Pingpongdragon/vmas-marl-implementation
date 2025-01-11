### Scheduler Class ###
import numpy as np

class Scheduler:
    "Used to manage annealing/decaying variables over time."
    def __init__(self, start, end, duration, fraction=0.9, log_decay=False):
        self.start = start
        self.end = end
        self.total_duration = duration
        self.duration = int(duration * fraction)
        self.log_decay = log_decay
        self.step = 0

        if log_decay:
            assert start > 0 and end > 0 and start > end, "Start and end must be > 0 and start > end."
            self.start = np.log(start)
            self.end = np.log(end)
        else:
            self.start = start
            self.end = end
        
    def get(self):
        "Gets current value without incrementing step counter."
        if self.step < self.duration:
            t = self.step / self.duration
            if self.log_decay:
                current_value = np.exp(self.start + (self.end - self.start) * t)
            else:
                current_value = self.start + (self.end - self.start) * t
        else:
            current_value = np.exp(self.end) if self.log_decay else self.end
        return current_value

    def __call__(self):
        "Gets current value and increments step counter."
        current_value = self.get()
        self.step += 1
        return current_value