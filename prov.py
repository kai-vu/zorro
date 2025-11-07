import os, re, json, mimetypes, hashlib, subprocess, logging, sys, inspect
from pathlib import Path
from datetime import datetime, timezone

import rdflib
from rdflib import XSD, RDFS, RDF
from rdflib.namespace import DCTERMS as DCT

PROV = rdflib.Namespace("http://www.w3.org/ns/prov#")

def add_dict(g, mapping: dict):
    for s, po in mapping.items():
        for p, o in po.items():
            pred = RDF.type if p == "a" else p
            objs = o if isinstance(o, (list, tuple, set)) else [o]
            for obj in objs:
                g.add((s, pred, obj))



def _caller_script():
    # 1) normal CLI
    mod = sys.modules.get('__main__')
    if getattr(mod, '__file__', None):
        return Path(mod.__file__).resolve()

    # 2) fallback: argv[0]
    if sys.argv and sys.argv[0]:
        p = Path(sys.argv[0])
        if p.exists():
            return p.resolve()

    # 3) last resort: outermost non-library frame
    for f in reversed(inspect.stack()):
        p = Path(f.filename)
        if p.name != 'prov.py' and p.suffix in {'.py', ''}:
            return p.resolve()

    return Path('unknown')


class Provenance:
    """Context manager for a named graph with lightweight PROV around open file handles."""

    def __init__(self, base: str, name: str, files: list, out_dir: str):
        self.base = rdflib.Namespace(base) if isinstance(base, str) else base
        self.name = name
        self.files = list(files)
        self.out_dir = out_dir

        self.cg = rdflib.Dataset()
        self.D = self.cg.default_graph
        self.cg.bind("", self.base)
        self.cg.bind("prov", PROV)
        self.cg.bind("dcterms", DCT)

        self.graph_iri = self.base[f"graph/{name}"]
        self.gx = self.cg.get_context(self.graph_iri)

        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.script = _caller_script()
        self.activity = self.base[f"run/{self.run_id}"]
        self.agent = self.base[f"agent/{self.script.name}"]
        self.plan = self.base[f"plan/{self.script.name}"]

        # git info
        self.commit = self._safe_cmd(["git", "rev-parse", "HEAD"])
        self.origin = self._safe_cmd(["git", "config", "--get", "remote.origin.url"])

        self.t0 = None
        self.t1 = None

    def __enter__(self):
        self.t0 = datetime.now(timezone.utc)
        return self  # use .gx to add triples

    def __exit__(self, exc_type, exc, tb):
        self.t1 = datetime.now(timezone.utc)
        # core PROV
        add_dict(self.D, {
            self.activity: {
                "a": PROV.Activity,
                PROV.startedAtTime: rdflib.Literal(self.t0.isoformat(), datatype=XSD.dateTime),
                PROV.endedAtTime:   rdflib.Literal(self.t1.isoformat(), datatype=XSD.dateTime),
                PROV.wasAssociatedWith: self.agent,
                PROV.used: self.plan,
            },
            self.plan: {
                "a": PROV.Plan,
                RDFS.label: rdflib.Literal(self.script.name),
                DCT.hasVersion: [rdflib.Literal(self.commit)] if self.commit else [],
                DCT.source: [rdflib.URIRef(self.origin)] if self.origin else [],
            },
            self.agent: {"a": PROV.SoftwareAgent, RDFS.label: rdflib.Literal(self.script.name)},
            self.graph_iri: {
                "a": PROV.Entity,
                PROV.wasGeneratedBy: self.activity,
                PROV.wasAttributedTo: self.agent,
                PROV.generatedAtTime: rdflib.Literal(self.t1.isoformat(), datatype=XSD.dateTime),
            },
        })

        # file inputs/outputs
        for fh in self.files:
            try:
                path = Path(fh.name)
                mode = getattr(fh, "mode", "")
                if "w" in mode or "a" in mode:
                    try: fh.flush()
                    except Exception: pass
                self._describe_file(path, mode)
            except Exception:
                continue

        # link graph to inputs (primary sources)
        for path, mode in self._classify_paths().items():
            src_iri = self._iri_for_path(path, mode)
            if "r" in mode and "w" not in mode:
                add_dict(self.D, {
                    self.graph_iri: {
                        PROV.wasDerivedFrom: src_iri,
                        PROV.hadPrimarySource: src_iri,
                    },
                    self.activity: {PROV.used: src_iri},
                })
            else:
                add_dict(self.D, {src_iri: {PROV.wasGeneratedBy: self.activity}})

        # write out
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        out_path = f"{self.out_dir}/{self.name}.trig"
        logging.info(f"Writing {out_path}")
        self.cg.serialize(out_path, format="trig")

        return False  # do not suppress exceptions

    # ---- internals ----
    def _safe_cmd(self, argv):
        try:
            return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None

    def _classify_paths(self):
        """Return {Path: mode} for known file handles."""
        m = {}
        for fh in self.files:
            try:
                p = Path(fh.name)
                m[p] = getattr(fh, "mode", "")
            except Exception:
                continue
        return m

    def _iri_for_path(self, path: Path, mode: str):
        kind = "src" if "r" in mode and "w" not in mode else "out"
        return self.base[f"{kind}/{path.as_posix()}"]

    def _describe_file(self, path: Path, mode: str):
        """Emit DCT metadata for a path and attach basic PROV."""
        src_iri = self._iri_for_path(path, mode)
        mtype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size if path.exists() else 0
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None

        items = {
            "a": PROV.Entity,
            DCT.format: rdflib.Literal(mtype),
            DCT.extent: rdflib.Literal(size, datatype=XSD.integer),
        }
        if mtime:
            items[DCT.modified] = rdflib.Literal(mtime, datatype=XSD.dateTime)

        # hash only for readable inputs to avoid partial-output hashing
        if "r" in mode and path.exists():
            try:
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                items[DCT.identifier] = rdflib.Literal(f"sha256:{sha}")
            except Exception:
                pass

        add_dict(self.D, {src_iri: items})

    # expose named graph for user code
    @property
    def gx(self):
        return self._gx

    @gx.setter
    def gx(self, value):
        self._gx = value
