import pytest

from tushare_a_fundamentals.common import (
    RetryExhaustedError,
    RetryPolicy,
    call_with_retry,
)

pytestmark = pytest.mark.unit


def test_call_with_retry_succeeds_after_retry():
    attempts = {"count": 0}

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient error")
        return "ok"

    policy = RetryPolicy(max_retries=2, base_delay=0.01, jitter=0)
    retries_log: list[tuple[int, str]] = []

    result = call_with_retry(
        flaky_call,
        policy=policy,
        on_retry=lambda attempt, exc, wait: retries_log.append((attempt, str(exc))),
        sleep_func=lambda _: None,
    )

    assert result == "ok"
    assert retries_log and retries_log[0][0] == 1
    assert attempts["count"] == 2


def test_call_with_retry_stops_on_fatal_exception():
    def fatal_call():
        raise ValueError("参数错误")

    policy = RetryPolicy(max_retries=3)
    with pytest.raises(ValueError):
        call_with_retry(fatal_call, policy=policy, sleep_func=lambda _: None)


def test_call_with_retry_exhaustion_raises_retry_error():
    policy = RetryPolicy(max_retries=1, base_delay=0.01, jitter=0)

    with pytest.raises(RetryExhaustedError) as excinfo:
        call_with_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("still failing")),
            policy=policy,
            sleep_func=lambda _: None,
        )

    assert isinstance(excinfo.value.last_exception, RuntimeError)
