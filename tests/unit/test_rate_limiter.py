import pytest

from tushare_a_fundamentals.downloader import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_enforces_window(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(
        "tushare_a_fundamentals.downloader.time.time", clock.time
    )
    monkeypatch.setattr(
        "tushare_a_fundamentals.downloader.time.sleep", clock.sleep
    )

    limiter = RateLimiter(max_per_minute=2)

    limiter.wait()
    limiter.wait()
    limiter.wait()

    assert clock.sleeps
    assert pytest.approx(clock.sleeps[0], rel=1e-3) == 60.1
    assert clock.now >= 60.1


def test_rate_limiter_no_limit(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(
        "tushare_a_fundamentals.downloader.time.time", clock.time
    )
    monkeypatch.setattr(
        "tushare_a_fundamentals.downloader.time.sleep", clock.sleep
    )

    limiter = RateLimiter(max_per_minute=0)
    for _ in range(5):
        limiter.wait()

    assert clock.sleeps == []
