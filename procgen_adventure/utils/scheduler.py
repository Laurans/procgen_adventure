import numpy as np


class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=0):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1

        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=0):
        val = self.bound(self.current + self.inc * steps, self.end)
        return val


class StepDecaySchedule:
    def __init__(self, val, rate, every_n_steps):
        self.val = val
        self.rate = rate
        self.every_n_steps = every_n_steps

    def __call_(self, steps=0):
        if steps != 0 and steps % self.every_n_steps == 0:
            val = self.val * self.rate
            self.val = val

        return self.val


class ExponentialDecaySchedule:
    def __init__(self, val, decay_constant=1):
        self.val = val
        self.k = decay_constant

    def __call__(self, steps=0):
        return self.val * np.exp(-self.k * steps)


class TemperatureDecaySchedule:
    def __init__(self, val, decay_constant):
        self.val = val
        self.k = decay_constant

    def __call__(self, steps=1):
        return self.val / (1 + self.k * steps)
