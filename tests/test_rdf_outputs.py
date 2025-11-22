from __future__ import annotations

import rdflib
from makeprov import OutPath

from make_rdf import (
    build_extractions_regex_graph,
    build_functions_graph,
    build_parts_graph,
    build_troubleshooting_graph,
)


def _assert_trig_build(fn, path):
    dataset = fn(trig_out=OutPath(str(path)))
    assert dataset is not None
    parsed = rdflib.ConjunctiveGraph()
    parsed.parse(path, format="trig")
    assert len(list(parsed.contexts())) >= 1


def test_troubleshooting_trig(tmp_path):
    _assert_trig_build(
        build_troubleshooting_graph,
        tmp_path / "troubleshooting.trig",
    )


def test_parts_trig(tmp_path):
    _assert_trig_build(build_parts_graph, tmp_path / "parts.trig")


def test_functions_trig(tmp_path):
    _assert_trig_build(build_functions_graph, tmp_path / "functions.trig")


def test_extractions_regex_trig(tmp_path):
    _assert_trig_build(
        build_extractions_regex_graph,
        tmp_path / "extractions.regex.trig",
    )
