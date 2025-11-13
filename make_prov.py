"""Make-like task registry that emits provenance

Usage:

```
import rdflib
from pathlib import Path

from make_prov import rule, build

@rule(
    target="build/graph.ttl",
    base_iri="http://example.org/",
    deps=["data/input.ttl", "data/config.json"],
    name="output_graph",
    prov_dir="prov",      # or prov_path="prov/custom_name.trig"
)
def build_graph():
    g = rdflib.Graph()
    g.parse("data/input.ttl", format="turtle")
    # ... transform g ...
    out = Path("build/graph.ttl")
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(out, format="turtle")
    return g  # becomes the named graph

if __name__ == "__main__":
    build("build/graph.ttl")
```
    
"""

import logging
import subprocess
import sys
import inspect
import mimetypes
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import rdflib
from rdflib import RDF, RDFS, XSD
from rdflib.namespace import DCTERMS as DCT

PROV = rdflib.Namespace("http://www.w3.org/ns/prov#")

RULES = {}

def _caller_script():
    """Best-effort guess of the calling script path."""
    mod = sys.modules.get("__main__")
    if getattr(mod, "__file__", None):
        return Path(mod.__file__).resolve()

    if sys.argv and sys.argv[0]:
        p = Path(sys.argv[0])
        if p.exists():
            return p.resolve()

    for f in reversed(inspect.stack()):
        p = Path(f.filename)
        if p.suffix in {".py", ""}:
            return p.resolve()

    return Path("unknown")

def _safe_cmd(argv):
    try:
        return subprocess.run(
            argv, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return None


def needs_update(outputs, deps):
    """Return True if outputs missing or older than any dependency."""
    out_paths = [Path(o) for o in outputs]
    dep_paths = [Path(d) for d in deps]

    if not out_paths:
        return True

    if any(not o.exists() for o in out_paths):
        return True

    oldest_out = min(o.stat().st_mtime for o in out_paths)
    dep_times = [d.stat().st_mtime for d in dep_paths if d.exists()]
    if not dep_times:
        # No existing deps to compare; assume up to date
        return False

    newest_dep = max(dep_times)
    return newest_dep > oldest_out


def build(target, _seen=None):
    """Recursively build target after its dependencies, if needed."""
    if _seen is None:
        _seen = set()
    if target in _seen:
        raise RuntimeError(f"Cycle in build graph at {target!r}")
    _seen.add(target)

    rule = RULES[target]

    # Build dependency targets first
    for dep in rule["deps"]:
        if dep in RULES:
            build(dep, _seen)

    # Run rule only if needed
    if needs_update(rule["outputs"], rule["deps"]):
        rule["func"]()


# ----------------------------------------------------------------------
# Simplified PROV with named data graph
# ----------------------------------------------------------------------


def _describe_file(g, base, path, kind):
    """
    Add basic PROV + DCT metadata for a file and return its IRI.

    kind: "src" for inputs, "out" for outputs.
    """
    p = Path(path)
    iri = base[f"{kind}/{p.as_posix()}"]

    mtype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    size = p.stat().st_size if p.exists() else 0
    mtime = (
        datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        if p.exists()
        else None
    )

    g.add((iri, RDF.type, PROV.Entity))
    g.add((iri, DCT.format, rdflib.Literal(mtype)))
    g.add((iri, DCT.extent, rdflib.Literal(size, datatype=XSD.integer)))
    if mtime:
        g.add((iri, DCT.modified, rdflib.Literal(mtime, datatype=XSD.dateTime)))

    # Hash only for inputs
    if kind == "src" and p.exists():
        try:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            g.add((iri, DCT.identifier, rdflib.Literal(f"sha256:{sha}")))
        except Exception:
            pass

    return iri


def _write_provenance_dataset(
    base_iri,
    name,
    prov_path,
    deps,
    outputs,
    t0,
    t1,
    data_graph=None,
    success=True,
):
    """
    Build a Dataset with:
      - default graph: PROV metadata
      - named graph:   data_graph (if provided)
    and serialize as Trig to prov_path.
    """
    base = rdflib.Namespace(base_iri)
    ds = rdflib.Dataset()
    D = ds.default_context

    ds.bind("", base)
    ds.bind("prov", PROV)
    ds.bind("dcterms", DCT)

    run_id = t0.strftime("%Y%m%dT%H%M%S")
    script = _caller_script()

    activity = base[f"run/{name}/{run_id}"]
    agent = base[f"agent/{script.name}"]
    graph_iri = base[f"graph/{name}"]

    commit = _safe_cmd(["git", "rev-parse", "HEAD"])
    origin = _safe_cmd(["git", "config", "--get", "remote.origin.url"])

    # Activity
    D.add((activity, RDF.type, PROV.Activity))
    t0_n3 = rdflib.Literal(t0.isoformat(), datatype=XSD.dateTime)
    D.add((activity, PROV.startedAtTime, t0_n3))
    t1_n3 = rdflib.Literal(t1.isoformat(), datatype=XSD.dateTime)
    D.add((activity, PROV.endedAtTime, t1_n3))

    # Agent (the script)
    D.add((agent, RDF.type, PROV.SoftwareAgent))
    D.add((agent, RDFS.label, rdflib.Literal(script.name)))
    if commit:
        D.add((agent, DCT.hasVersion, rdflib.Literal(commit)))
    if origin:
        D.add((agent, DCT.source, rdflib.URIRef(origin)))
    D.add((activity, PROV.wasAssociatedWith, agent))

    # Named data graph as PROV Entity
    if data_graph is not None:
        gx = ds.get_context(graph_iri)
        for triple in data_graph:
            gx.add(triple)

        D.add((graph_iri, RDF.type, PROV.Entity))
        D.add((graph_iri, PROV.wasGeneratedBy, activity))
        D.add((graph_iri, PROV.wasAttributedTo, agent))

        D.add((graph_iri, PROV.generatedAtTime, t1_n3))

    # Inputs
    for d in deps:
        p = Path(d)
        if not p.exists():
            continue
        src = _describe_file(D, base, p, "src")
        D.add((activity, PROV.used, src))

    # Outputs (not including prov_path itself)
    for o in outputs:
        p = Path(o)
        if not p.exists():
            continue
        ent = _describe_file(D, base, p, "out")
        D.add((ent, PROV.wasGeneratedBy, activity))

    if not success:
        D.add((activity, RDFS.comment, rdflib.Literal("task failed")))

    prov_path = Path(prov_path)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing provenance dataset %s", prov_path)
    ds.serialize(prov_path, format="trig")


# ----------------------------------------------------------------------
# Decorator that registers a rule and wraps it with PROV + named graph
# ----------------------------------------------------------------------


def rule(
    target,
    base_iri,
    deps=(),
    outputs=None,
    name=None,
    prov_dir="prov",
    prov_path=None,
):
    """
    Decorator for a Make-like rule with automatic provenance.

    target:    main output file path (string)
    deps:      iterable of file paths (strings); some may also be other targets
    outputs:   iterable of output file paths; defaults to [target]
    base_iri:  base IRI for PROV entities and the data named graph
    name:      logical name for this rule/run; defaults to stem(target)
    prov_dir:  directory where provenance .trig is written if prov_path not set
    prov_path: full path for provenance .trig; overrides prov_dir/name.trig

    If the wrapped function returns an rdflib.Graph (but not Dataset),
    that graph is stored as a named graph at base_iri + "graph/{name}".
    """
    outputs = list(outputs) if outputs is not None else [target]
    deps = list(deps)
    name = name or Path(target).stem
    if prov_path is None:
        prov_path = str(Path(prov_dir) / f"{name}.trig")

    def decorator(func):
        def wrapped():
            t0 = datetime.now(timezone.utc)
            exc = None
            data_graph = None
            result = None

            try:
                result = func()

                # If user returned a Graph, attach it as named graph
                if isinstance(result, rdflib.Graph) and not isinstance(
                    result, rdflib.Dataset
                ):
                    data_graph = result

                return result
            except Exception as e:
                exc = e
                raise
            finally:
                t1 = datetime.now(timezone.utc)
                try:
                    _write_provenance_dataset(
                        base_iri=base_iri,
                        name=name,
                        prov_path=prov_path,
                        deps=deps,
                        outputs=outputs,
                        t0=t0,
                        t1=t1,
                        data_graph=data_graph,
                        success=exc is None,
                    )
                except Exception as prov_exc:
                    logging.warning(
                        "Failed to write provenance for %s: %s",
                        name,
                        prov_exc,
                    )

        RULES[target] = {
            "deps": deps,
            "outputs": outputs,
            "func": wrapped,
        }
        return wrapped

    return decorator
