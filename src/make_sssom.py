"""Generate SSSOM TSV and RDF outputs for part links."""

from __future__ import annotations

import logging
from pathlib import Path

import defopt

from make_prov import build, rule
from zorro_wp2 import load_default_config
from zorro_wp2.sssom import build_sssom


BASE_IRI = "https://w3id.org/zorro/pipeline/"
METHOD = "regex"
TSV_TARGET = f"part-links/part-links-{METHOD}.sssom.tsv"
RDF_TARGET = f"generated-rdf/part-links-sssom-{METHOD}.trig"

_CLI_VERBOSE = False


@rule(
    target=TSV_TARGET,
    outputs=(TSV_TARGET, RDF_TARGET),
    deps=(f"part-links/part-links-{METHOD}.tsv",),
    base_iri=BASE_IRI,
    name="make-sssom",
    prov_dir="prov",
    tracked_packages=("rdflib",),
    metadata={"mapping_method": "sssom-from-tfidf"},
    graph_name=f"graph/part-links-sssom/{METHOD}",
)
def make_sssom() -> tuple[Path, Path]:
    """Create SSSOM TSV and RDF outputs."""
    config = load_default_config()
    logging.debug("Building SSSOM artefacts verbose=%s", _CLI_VERBOSE)
    return build_sssom(config, method=METHOD, verbose=_CLI_VERBOSE)


def main(*, overwrite: bool = False, verbose: bool = False) -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    global _CLI_VERBOSE
    _CLI_VERBOSE = verbose

    if overwrite:
        Path(TSV_TARGET).unlink(missing_ok=True)
        Path(RDF_TARGET).unlink(missing_ok=True)

    build(TSV_TARGET, force=overwrite or verbose)


if __name__ == "__main__":
    defopt.run(main)
