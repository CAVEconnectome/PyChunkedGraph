"""Tests for pychunkedgraph.meshing.manifest.v2."""

from pychunkedgraph.meshing.manifest.v2 import assemble, to_v2_groups


class TestToV2Groups:
    def test_prepend_matches_v1_grouped_by_type(self):
        """With prepend_seg_ids each fragment is v1's ``~<segid>:<frag>``; grouping
        is by the raw ~ (initial/sharded) vs no-~ (dynamic/whole-file)."""
        node_ids = [386, 529]
        frags = ["~5/774017-0.shard:74809813:4532", "529:0:90112-98304"]
        ini, dyn = to_v2_groups(node_ids, frags, prepend_seg_ids=True)
        assert ini == ["~386:~5/774017-0.shard:74809813:4532"]
        assert dyn == ["~529:529:0:90112-98304"]

    def test_no_prepend_keeps_raw(self):
        ini, dyn = to_v2_groups(
            [386, 529], ["~a.shard:1:2", "529:0:bbox"], prepend_seg_ids=False
        )
        assert ini == ["~a.shard:1:2"]
        assert dyn == ["529:0:bbox"]

    def test_assemble_groups_under_bucket(self):
        resp = assemble(
            "gs://b/initial", "gs://b/dynamic", ["~386:~a.shard:1:2"], ["~529:529:0:y"]
        )
        assert resp["fragments"]["gs://b/initial"]["fragments"] == ["~386:~a.shard:1:2"]
        assert resp["manifest_version"] == 2
