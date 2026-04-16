"""Tests for pychunkedgraph.meshing.manifest.utils"""

from pychunkedgraph.meshing.manifest.utils import _del_none_keys


class TestDelNoneKeys:
    def test_removes_none_values(self):
        d = {"a": 1, "b": None, "c": 3}
        result, removed = _del_none_keys(d)
        assert result == {"a": 1, "c": 3}
        assert set(removed) == {"b"}

    def test_no_none_values(self):
        d = {"a": 1, "b": 2}
        result, removed = _del_none_keys(d)
        assert result == {"a": 1, "b": 2}
        assert removed == []

    def test_all_none_values(self):
        d = {"a": None, "b": None}
        result, removed = _del_none_keys(d)
        assert result == {}
        assert set(removed) == {"a", "b"}

    def test_empty_dict(self):
        result, removed = _del_none_keys({})
        assert result == {}
        assert removed == []

    def test_original_not_mutated(self):
        d = {"a": 1, "b": None}
        _del_none_keys(d)
        assert d == {"a": 1, "b": None}

    def test_falsy_values_removed(self):
        """The function uses `if v:` so falsy values like 0, [], '' are also removed."""
        d = {"a": 0, "b": [], "c": "", "d": "valid"}
        result, removed = _del_none_keys(d)
        assert result == {"d": "valid"}
        assert set(removed) == {"a", "b", "c"}
