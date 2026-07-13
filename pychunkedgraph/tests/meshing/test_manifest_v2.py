"""Tests for pychunkedgraph.meshing.manifest.v2."""

from pychunkedgraph.meshing.manifest.v2 import assemble, to_v2_groups


class TestToV2Groups:
    def test_partitions_by_tilde_marker(self):
        ini, dyn = to_v2_groups(
            ["~5/774017-0.shard:74809813:4532", "529735906171415924:0:90112-98304"]
        )
        assert ini == ["~5/774017-0.shard:74809813:4532"]
        assert dyn == ["529735906171415924:0:90112-98304"]

    def test_keeps_fragments_verbatim(self):
        """The leading ~ is the client's sharded/whole-file dispatch flag, so
        neither group may strip or reshape the fragment string."""
        sharded = "~386882991802088261:5:173:425884686-0.shard:1"
        whole = "459631044782780793:0:98304-102400"
        ini, dyn = to_v2_groups([sharded, whole])
        assert ini == [sharded]
        assert dyn == [whole]

    def test_assemble_preserves_marker_under_bucket(self):
        resp = assemble("gs://b/initial", "gs://b/dynamic", ["~a.shard:1:2"], ["x:0:y"])
        assert resp["fragments"]["gs://b/initial"]["fragments"] == ["~a.shard:1:2"]
        assert resp["manifest_version"] == 2
