from types import SimpleNamespace

import pytest

import tushare_a_fundamentals.common as common
from tushare_a_fundamentals.common import ProPool

pytestmark = pytest.mark.unit


def test_pro_pool_round_robin_balances_clients(monkeypatch):
    schedules: dict[str, list[tuple[bool, float]]] = {}

    class DummyClient:
        def __init__(self, token: str, rate: int) -> None:
            self.token = token
            self.rate = rate
            schedules[token] = [(True, 0.0)] * 10
            self._schedule = schedules[token]
            self.pro = SimpleNamespace(query=lambda *args, **kwargs: token)

        def try_acquire(self, now: float) -> tuple[bool, float]:
            if self._schedule:
                return self._schedule.pop(0)
            return True, 0.0

        def set_rate(self, value: int) -> None:
            self.rate = value

    fake_time = SimpleNamespace(time=lambda: 0.0, sleep=lambda _wait: None)

    monkeypatch.setattr(common, "_TokenClient", DummyClient)
    monkeypatch.setattr(common, "time", fake_time)

    pool = ProPool(["tok-a", "tok-b", "tok-c"], per_token_rate=5)

    results = [pool.query("endpoint") for _ in range(6)]

    assert results == ["tok-a", "tok-b", "tok-c", "tok-a", "tok-b", "tok-c"]


def test_pro_pool_set_rate_propagates_to_clients(monkeypatch):
    rate_history: dict[str, list[int]] = {}

    class DummyClient:
        def __init__(self, token: str, rate: int) -> None:
            self.token = token
            rate_history[token] = [max(0, int(rate))]
            self.pro = SimpleNamespace(query=lambda *args, **kwargs: token)

        def try_acquire(self, now: float) -> tuple[bool, float]:
            return True, 0.0

        def set_rate(self, value: int) -> None:
            rate_history[self.token].append(value)

    monkeypatch.setattr(common, "_TokenClient", DummyClient)

    pool = ProPool(["tok-a", "tok-b"], per_token_rate=12)

    pool.set_rate(None)
    pool.set_rate(7)

    assert rate_history["tok-a"] == [12, 0, 7]
    assert rate_history["tok-b"] == [12, 0, 7]


def test_pro_pool_getattr_falls_back_to_query(monkeypatch):
    class DummyClient:
        def __init__(self, token: str, rate: int) -> None:
            self.token = token
            self.pro = SimpleNamespace(
                query=lambda name, *args, **kwargs: (self.token, name, args, kwargs)
            )

        def try_acquire(self, now: float) -> tuple[bool, float]:
            return True, 0.0

        def set_rate(self, value: int) -> None:
            pass

    monkeypatch.setattr(common, "_TokenClient", DummyClient)

    pool = ProPool(["tok-x"], per_token_rate=1)

    result = pool.some_missing_method(123, flag=True)

    assert result[0] == "tok-x"
    assert result[1] == "some_missing_method"
    assert result[2] == (123,)
    assert result[3] == {"flag": True}
