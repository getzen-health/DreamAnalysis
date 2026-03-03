import numpy as np
import pytest


def test_returns_raw_before_min_samples():
    from processing.eeg_processor import RunningNormalizer
    rn = RunningNormalizer()
    features = np.array([1.0, 2.0, 3.0])
    result = rn.normalize(features, "user1")
    np.testing.assert_array_equal(result, features)


def test_normalizes_after_30_samples():
    from processing.eeg_processor import RunningNormalizer
    rn = RunningNormalizer()
    # Feed 30 identical samples
    for _ in range(30):
        rn.normalize(np.array([5.0, 10.0, 3.0]), "user1")
    # 31st sample — std is near-zero, should return finite values (0s)
    result = rn.normalize(np.array([5.0, 10.0, 3.0]), "user1")
    assert np.all(np.isfinite(result))


def test_isolated_by_user():
    from processing.eeg_processor import RunningNormalizer
    rn = RunningNormalizer()
    for _ in range(30):
        rn.normalize(np.array([100.0, 200.0]), "user_a")
    # user_b has no history — should get raw features
    result = rn.normalize(np.array([1.0, 2.0]), "user_b")
    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))


def test_thread_safe():
    import threading
    from processing.eeg_processor import RunningNormalizer
    rn = RunningNormalizer()
    errors = []
    def worker():
        try:
            for _ in range(20):
                rn.normalize(np.random.randn(10), "shared")
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
