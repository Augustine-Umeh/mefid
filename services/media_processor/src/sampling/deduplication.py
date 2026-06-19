class SampleGuard:
    """Enforce a minimum time gap between sampled frames."""

    def __init__(self, min_gap_seconds: float = 1.0):
        self.min_gap = min_gap_seconds
        self._last_sampled_ts: float = -999.0

    def should_sample(self, timestamp: float) -> bool:
        return (timestamp - self._last_sampled_ts) >= self.min_gap

    def record(self, timestamp: float) -> None:
        self._last_sampled_ts = timestamp

    def try_sample(self, timestamp: float) -> bool:
        if self.should_sample(timestamp):
            self.record(timestamp)
            return True
        return False
