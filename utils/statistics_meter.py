r"""
Classes to handle statistics about values.
"""

from math import sqrt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageStddevMeter(AverageMeter):
    def __init__(self):
        super().__init__()
        self.sum_squared = 0
        self.stddev = 0

    def reset(self):
        super().reset()
        self.sum_squared = 0
        self.stddev = 0

    def update(self, val, n=1):
        super().update(val, n)
        self.sum_squared += n * val**2
        self.stddev = sqrt((self.sum_squared / self.count) - self.avg**2)

