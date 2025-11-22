from __future__ import annotations

import rdflib
from makeprov import OutPath
from sssom.parsers import parse_sssom_table

from link_parts import build_part_links_regex


def test_part_links_sssom(tmp_path):
    """SSSOM TSV should parse and emit triples."""
    tsv_path = tmp_path / "part-links.sssom.tsv"
    trig_path = tmp_path / "part-links.sssom.trig"

    build_part_links_regex(
        sssom_tsv=OutPath(str(tsv_path)),
        sssom_trig=OutPath(str(trig_path)),
    )

    msdf = parse_sssom_table(tsv_path)
    assert not msdf.df.empty

    required_cols = {
        "subject_id",
        "object_id",
        "predicate_id",
        "confidence",
        "mapping_tool",
        "mapping_date",
    }
    assert required_cols.issubset(msdf.df.columns)

    dataset = rdflib.ConjunctiveGraph()
    dataset.parse(trig_path, format="trig")
    assert len(list(dataset.contexts())) >= 1
