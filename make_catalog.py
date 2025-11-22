#!/usr/bin/env python3
"""
Generate a schema.org DataCatalog (JSON-LD) for a list of files.

Usage (defopt parses the function signature):
  python make_catalog.py data/*.ttl --base-url https://example.org/ --name "My data catalog" --queries queries/*
"""

from __future__ import annotations
import json
import mimetypes
from pathlib import Path
from typing import Iterable, Optional
import defopt

# Minimal RDF-focused media type fixes
_RDF_TYPES = {
    ".ttl":  "text/turtle",
    ".trig": "application/trig",
    ".nt":   "application/n-triples",
    ".nq":   "application/n-quads",
    ".jsonld": "application/ld+json",
    ".rdf":  "application/rdf+xml",
    ".xml":  "application/rdf+xml",
    ".rq": "application/sparql-query",
    ".sparql": "application/sparql-query",
}

def _guess_media_type(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in _RDF_TYPES:
        return _RDF_TYPES[ext]
    mt, _ = mimetypes.guess_type(p.name)
    return mt or "application/octet-stream"

def _rel_url(base_url: str, file_path: Path) -> str:
    # Join base_url with a POSIX-style relative path
    rel = file_path.as_posix()
    if not base_url.endswith("/"):
        base_url += "/"
    return base_url + rel

def catalog(
    *files: Path,
    base_url: str,
    name: str,
    out: Optional[Path] = None,
    license: Optional[str] = None,
    queries: Optional[Iterable[Path]] = None,
    copy_ui: bool = False,
) -> None:
    """
    Create a JSON-LD DataCatalog from file paths.

    :param files: One or more file paths (relative paths are kept in URLs).
    :param base_url: Base URL where the files are hosted (e.g., https://user.github.io/repo/).
    :param name: Human-readable catalog name.
    :param out: Output file path (default: stdout).
    :param license: Optional license URL to apply to all datasets.
    :param queries: One or more sparql query file paths (relative paths are kept in URLs).
    :param copy_ui: Whether to copy query.html into the current directory or the parent directory of out.
    """
    if not files:
        raise SystemExit("No input files provided.")

    datasets = []
    for fp in files:
        p = Path(fp)
        identifier = p.stem
        media = _guess_media_type(p)
        content_url = _rel_url(base_url, p)

        ds = {
            "@type": "Dataset",
            "name": identifier,
            "identifier": identifier,
            "distribution": [{
                "@type": "DataDownload",
                "encodingFormat": media,
                "contentUrl": content_url,
            }],
        }
        if license:
            ds["license"] = license
        datasets.append(ds)

    # Build hasPart for queries
    has_part = []
    for q in (queries or []):
        q = Path(q)
        media = _guess_media_type(q)
        if media != "application/sparql-query":
            continue
        has_part.append({
            "@type": "SoftwareSourceCode",
            "name": q.stem.replace("_", " "),
            "programmingLanguage": "SPARQL",
            "encodingFormat": media,
            "contentUrl": _rel_url(base_url, q),
        })

    catalog = {
        "@context": "https://schema.org",
        "@type": "DataCatalog",
        "name": name,
        "url": base_url,
        "dataset": datasets,
    }
    if has_part:
        catalog["hasPart"] = has_part

    text = json.dumps(catalog, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text)

    if copy_ui:
        import shutil, sys

        dest: str = "query.html"
        if out:
            out = Path(out).parent / dest
        if not src:
            candidate = Path(__file__).with_name("query.html")
            if candidate.exists():
                src = candidate

        if not src or not src.exists():
            raise SystemExit("query.html not found in source tree")

        shutil.copyfile(src, dest)
        print(f"Copied {src} -> {dest}", file=sys.stderr)

if __name__ == "__main__":
    defopt.run(catalog)
