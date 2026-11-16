"""Tests for pychunkedgraph.meshing.manifest.v2."""

import pytest

from pychunkedgraph.meshing.manifest.v2 import (
    MANIFEST_VERSION,
    assemble,
    served_manifest_version,
    to_v2_groups,
)

ASK = "application/x.cave;manifest_version={}"


class TestServedManifestVersion:
    @pytest.mark.parametrize(
        "accept, served",
        [
            (None, 1),
            ("application/json", 1),
            (ASK.format("x"), 1),
            (ASK.format(0), 1),
            (ASK.format(1), 1),
            (ASK.format(MANIFEST_VERSION), MANIFEST_VERSION),
            (ASK.format(MANIFEST_VERSION + 1), MANIFEST_VERSION),
            (f"application/json, {ASK.format(MANIFEST_VERSION)}", MANIFEST_VERSION),
        ],
    )
    def test_an_ask_for_the_newest_or_past_it_gets_the_newest_and_any_other_v1(
        self, accept, served
    ):
        """No ask, an unparseable one or one below the newest gets v1."""
        assert served_manifest_version(accept) == served


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
            "gs://b/initial", ["386:a.shard:1:2"], {"gs://b/dynamic": ["529:0:y"]}
        )
        assert resp["fragments"]["gs://b/initial"] == ["386:a.shard:1:2"]
        assert resp["fragments"]["gs://b/dynamic"] == ["529:0:y"]
        assert resp["manifest_version"] == 2

    def test_assemble_keeps_each_dynamic_dir_under_its_own_bucket(self):
        """A copy's rows from its source's dir stay apart from its own."""
        dynamic = {"gs://src/dynamic": ["529:0:y"], "gs://b/dynamic": ["530:0:z"]}
        resp = assemble("gs://b/initial", [], dynamic)
        assert resp["fragments"]["gs://src/dynamic"] == ["529:0:y"]
        assert resp["fragments"]["gs://b/dynamic"] == ["530:0:z"]

    def test_assemble_omits_a_group_with_no_fragments(self):
        dynamic = {"gs://src/dynamic": [], "gs://b/dynamic": ["529:0:y"]}
        resp = assemble("gs://b/initial", [], dynamic)
        assert list(resp["fragments"]) == ["gs://b/dynamic"]
