"""Tests for pychunkedgraph.meshing.manifest.v2."""

from pychunkedgraph.meshing.manifest.v2 import assemble, to_v2_groups


class TestToV2Groups:
    def test_a_verified_row_gains_its_seg_id(self):
        """A verified v1 row names no segment, so v2 puts its seg id in front; a dynamic
        fragment already opens with its own and passes through."""
        frags = ["~5/774017-0.shard:74809813:4532", "529:0:90112-98304"]
        ini, dyn = to_v2_groups([386, 529], frags, verified=True)
        assert ini == ["386:5/774017-0.shard:74809813:4532"]
        assert dyn == ["529:0:90112-98304"]

    def test_a_speculative_row_keeps_one_seg_id_and_drops_the_shard_extension(self):
        """Only a row carrying its byte range names a ``.shard`` file."""
        segid = 458522737061959004
        frags = [f"~{segid}:6:458522737061658624:95232-0.shard:85", "529:0:90112-98304"]
        ini, dyn = to_v2_groups([segid, 529], frags, verified=False)
        assert ini == [f"{segid}:6:458522737061658624:95232-0:85"]
        assert dyn == ["529:0:90112-98304"]

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
