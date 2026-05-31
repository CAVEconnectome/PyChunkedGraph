"""Tests for pychunkedgraph.profiler.hierarchical (fold-on-record profiler).

Behavior, not constants: a disabled profiler is a true no-op; profile() folds
same-path blocks; the hot path spawns no sampler thread (the regression that
corrupted stage timings); from_blocks composes across processes; the profiler
pickles (sv_split persists it); and the existing consumers (print_report /
metrics_report / read-only blocks) keep working.
"""

import io
import pickle
import threading
from contextlib import redirect_stdout
from unittest import mock

import psutil

from pychunkedgraph.profiler import HierarchicalProfiler


def _loads(prof):
    return prof.blocks


class TestNoOpWhenDisabled:
    def test_no_memory_info_and_no_blocks(self):
        prof = HierarchicalProfiler(enabled=False)
        calls = {"n": 0}
        real = psutil.Process.memory_info

        def counting(self):
            calls["n"] += 1
            return real(self)

        with mock.patch.object(psutil.Process, "memory_info", counting):
            for _ in range(500):
                with prof.profile("x", sampled_rss=True):
                    pass
                prof.sample_rss()
        assert calls["n"] == 0
        assert prof.blocks == [] and not prof.timings


class TestLightweight:
    def test_substage_blocks_spawn_no_thread(self):
        # Regression guard: with with_rss_default=False (how the stitch worker is
        # configured) a bare profile() block is timing-only — no sampler thread
        # per call. A per-fragment thread would corrupt the stage timings.
        prof = HierarchicalProfiler(enabled=True, with_memory=False, with_rss=False)
        base = threading.active_count()
        seen = []
        for name in ("decode", "transform", "merge", "encode"):
            with prof.profile(name):
                seen.append(threading.active_count())
        with prof.profile("stitch", sampled_rss=True):
            seen.append(threading.active_count())
        assert all(c == base for c in seen), (seen, base)

    def test_sampled_rss_is_threadless(self):
        prof = HierarchicalProfiler(enabled=True, with_rss=False)
        rss = iter([100, 700])
        with mock.patch.object(prof, "sample_rss", lambda: next(rss)):
            base = threading.active_count()
            with prof.profile("s", sampled_rss=True):
                during = threading.active_count()
        assert during == base
        b = prof.blocks[0]
        assert b.rss_start_bytes == 100 and b.rss_peak_bytes == 700


class TestFoldOnRecord:
    def test_same_path_folds_to_one_block(self):
        prof = HierarchicalProfiler(enabled=True, with_memory=False, with_rss=False)
        perf = iter([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
        rss = iter([100, 500, 100, 200, 100, 400])
        with mock.patch("time.perf_counter", lambda: next(perf)), mock.patch.object(
            prof, "sample_rss", lambda: next(rss)
        ):
            for _ in range(3):
                with prof.profile("enc", sampled_rss=True):
                    pass
        blocks = prof.blocks
        assert len(blocks) == 1
        b = blocks[0]
        assert abs(b.elapsed_s - 6.0) < 1e-9  # summed
        assert b.rss_peak_bytes == 500  # max
        assert b.rss_start_bytes == 100  # min-nonzero
        assert b.call_count == 3  # 1 per call, summed
        assert prof.call_counts["enc"] == 3

    def test_rss_off_blocks_do_not_pin_start_to_zero(self):
        prof = HierarchicalProfiler(enabled=True, with_rss=False)
        rss = iter([300, 700])  # only the sampled_rss block samples
        seq = iter([0.0, 0.1, 0.0, 0.1])
        with mock.patch("time.perf_counter", lambda: next(seq)), mock.patch.object(
            prof, "sample_rss", lambda: next(rss)
        ):
            with prof.profile("s"):  # no rss -> 0/0
                pass
            with prof.profile("s", sampled_rss=True):  # 300/700
                pass
        b = prof.blocks[0]
        assert b.rss_start_bytes == 300 and b.rss_peak_bytes == 700

    def test_block_count_bounded_by_distinct_paths(self):
        prof = HierarchicalProfiler(enabled=True, with_rss=False)
        for _ in range(1000):
            with prof.profile("a"):
                pass
            with prof.profile("b"):
                pass
        assert len(prof.blocks) == 2


class TestFromBlocksComposes:
    def test_cross_process_merge(self):
        a = HierarchicalProfiler(enabled=True, with_rss=False)
        sa = iter([0.0, 2.0])
        ra = iter([100, 400])
        with mock.patch("time.perf_counter", lambda: next(sa)), mock.patch.object(
            a, "sample_rss", lambda: next(ra)
        ):
            with a.profile("m", sampled_rss=True):
                pass
        b = HierarchicalProfiler(enabled=True, with_rss=False)
        sb = iter([0.0, 3.0])
        rb = iter([200, 900])
        with mock.patch("time.perf_counter", lambda: next(sb)), mock.patch.object(
            b, "sample_rss", lambda: next(rb)
        ):
            with b.profile("m", sampled_rss=True):
                pass
        merged = HierarchicalProfiler.from_blocks(a.blocks + b.blocks).blocks[0]
        assert abs(merged.elapsed_s - 5.0) < 1e-9  # summed
        assert merged.rss_peak_bytes == 900  # max
        assert merged.rss_start_bytes == 100  # min
        assert merged.call_count == 2


class TestConsumersIntact:
    def test_pickle_round_trip_drops_proc(self):
        prof = HierarchicalProfiler(enabled=True)
        prof.sample_rss()
        assert prof._proc is not None
        restored = pickle.loads(pickle.dumps(prof))
        assert restored._proc is None

    def test_print_report_total_matches_timings(self):
        prof = HierarchicalProfiler(enabled=True, with_rss=False)
        seq = iter([0.0, 0.5, 0.0, 0.7])
        with mock.patch("time.perf_counter", lambda: next(seq)):
            with prof.profile("a"):
                pass
            with prof.profile("b"):
                pass
        buf = io.StringIO()
        with redirect_stdout(buf):
            prof.print_report()
        assert "PROFILER REPORT" in buf.getvalue()
        assert abs(sum(sum(t) for t in prof.timings.values()) - 1.2) < 1e-9

    def test_metrics_report_renders(self):
        prof = HierarchicalProfiler(enabled=True, with_rss=False)
        with prof.profile("a"):
            pass
        buf = io.StringIO()
        with redirect_stdout(buf):
            prof.metrics_report()
        assert "metrics report" in buf.getvalue()

    def test_reports_nest_paths_into_level_columns(self):
        # nested paths (stitch.decode) render in separate L0/L1 columns, the same
        # tree layout for both reports; each table ends with a blank line.
        prof = HierarchicalProfiler(enabled=True, with_rss=False)
        with prof.profile("stitch"):
            with prof.profile("decode"):
                pass
        for render in (
            lambda: prof.metrics_report(),
            lambda: HierarchicalProfiler.percentile_report(prof.blocks),
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                render()
            out = buf.getvalue()
            header = next(ln for ln in out.splitlines() if ln.startswith("L0"))
            assert "L0" in header and "L1" in header, header
            # the nested child's leaf name appears (not the dotted path).
            assert "decode" in out and "stitch.decode" not in out
            # the table block is followed by a blank line.
            assert "\n\n" in out, "table must be followed by a blank line"

    def test_blocks_is_read_only(self):
        prof = HierarchicalProfiler(enabled=True)
        try:
            prof.blocks = []
        except AttributeError:
            return
        raise AssertionError("blocks must be a read-only property")
