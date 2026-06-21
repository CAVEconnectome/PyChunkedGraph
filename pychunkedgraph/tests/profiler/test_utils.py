"""Tests for pychunkedgraph.profiler.utils.cgroup_peak_bytes.

Behavior across environments: reads cgroup v2 ``memory.peak`` when present, falls
back to cgroup v1 ``memory.max_usage_in_bytes``, and returns ``None`` when neither
is readable (dev box / macOS) so callers degrade gracefully. The k8s pod-sizing
figure depends on this, so the v2/v1/none paths are all pinned.
"""

import builtins

import pytest

from pychunkedgraph.profiler.utils import cgroup_peak_bytes

V2 = "/sys/fs/cgroup/memory.peak"
V1 = "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"


def _fake_open(available):
    """Return an ``open`` replacement that serves ``available`` {path: text} and
    raises FileNotFoundError for anything else (mirrors a missing cgroup file)."""
    real_open = builtins.open

    def opener(path, *args, **kwargs):
        if path in available:
            import io

            return io.StringIO(available[path])
        if path in (V2, V1):
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    return opener


def test_reads_cgroup_v2_peak(monkeypatch):
    monkeypatch.setattr(builtins, "open", _fake_open({V2: "123456789\n"}))
    assert cgroup_peak_bytes() == 123456789


def test_falls_back_to_v1(monkeypatch):
    monkeypatch.setattr(builtins, "open", _fake_open({V1: "987654321\n"}))
    assert cgroup_peak_bytes() == 987654321


def test_v2_preferred_over_v1(monkeypatch):
    monkeypatch.setattr(builtins, "open", _fake_open({V2: "111\n", V1: "999\n"}))
    assert cgroup_peak_bytes() == 111


def test_none_when_no_cgroup(monkeypatch):
    monkeypatch.setattr(builtins, "open", _fake_open({}))
    assert cgroup_peak_bytes() is None


def test_none_on_garbage(monkeypatch):
    monkeypatch.setattr(builtins, "open", _fake_open({V2: "max\n"}))
    assert cgroup_peak_bytes() is None
