"""Tests for pychunkedgraph.meshing.manifest.v2."""

from pychunkedgraph.meshing.manifest.v2 import assemble, to_v2_groups


class TestToV2Groups:
    def test_return_seg_ids_prefixes_the_sharded_rows_alone(self):
        """A dynamic fragment already opens with its node id, so only a sharded row
        needs the prefix; neither kind carries v1's marker."""
        node_ids = [386, 529]
        frags = ["~5/774017-0.shard:74809813:4532", "529:0:90112-98304"]
        ini, dyn = to_v2_groups(node_ids, frags, return_seg_ids=True)
        assert ini == ["386:5/774017-0.shard:74809813:4532"]
        assert dyn == ["529:0:90112-98304"]

    def test_without_the_flag_no_row_carries_a_seg_id(self):
        ini, dyn = to_v2_groups(
            [386, 529], ["~a.shard:1:2", "529:0:bbox"], return_seg_ids=False
        )
        assert ini == ["a.shard:1:2"]
        assert dyn == ["529:0:bbox"]

    def test_assemble_maps_each_bucket_to_its_list(self):
        """A bucket maps straight to its fragments; metadata sits at the top level."""
        resp = assemble(
            "gs://b/initial", "gs://b/dynamic", ["386:a.shard:1:2"], ["529:0:y"]
        )
        assert resp["fragments"]["gs://b/initial"] == ["386:a.shard:1:2"]
        assert resp["fragments"]["gs://b/dynamic"] == ["529:0:y"]
        assert resp["manifest_version"] == 2

    def test_assemble_omits_a_group_with_no_fragments(self):
        resp = assemble("gs://b/initial", "gs://b/dynamic", [], ["529:0:y"])
        assert list(resp["fragments"]) == ["gs://b/dynamic"]
