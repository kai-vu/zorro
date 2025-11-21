"""CLI entrypoint for the end-to-end knowledge graph build workflow."""

from __future__ import annotations

import logging

from makeprov.config import main as makeprov_main

# Import rules so they register with makeprov
from link_parts import build_part_links_regex  # noqa: F401
from log_extract_regex import (  # noqa: F401
    extract_actions_regex,
    extract_problems_regex,
)
from log_extract_gpt import (  # noqa: F401
    extract_actions_chatgpt,
    extract_problems_chatgpt,
)
from log_extract_ner import (  # noqa: F401
    extract_problems_ner,
    train_ner_model,
)
from make_rdf import (  # noqa: F401
    build_extractions_gpt_graph,
    build_extractions_regex_graph,
    build_functions_graph,
    build_parts_graph,
    build_troubleshooting_graph,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Execute makeprov CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    makeprov_main()


if __name__ == "__main__":
    main()
