import csv, re, json, collections, logging, rdflib
from pathlib import Path
from datetime import datetime, timezone
from rdflib import XSD, RDFS, RDF
from rdflib.namespace import DC
import subprocess, hashlib, mimetypes

from prov import Provenance, add_dict

# ---------- helpers ----------
def iri_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", " ".join(s.split()[:10]).title())

# ---------- namespaces ----------
BASE = rdflib.Namespace("https://w3id.org/zorro#")

OUT_DIR = 'generated-trig'

# ---------- builders ----------
with open("pdf-extracted/troubleshooting.csv") as fh:
    with Provenance(BASE, 'troubleshooting', [fh], OUT_DIR) as prov:
        for row in csv.DictReader(fh):
            s_tr = BASE[iri_slug(row["TROUBLE"])]
            s_ca = BASE[iri_slug(row["PROBABLE CAUSE"])]
            s_re = BASE[iri_slug(row["REMEDY"])]
            add_dict(prov.gx, {
                s_tr: {RDFS.subClassOf: BASE.Problem,
                       RDFS.label: rdflib.Literal(row["TROUBLE"]),
                       BASE.hasCause: s_ca},
                s_ca: {RDFS.subClassOf: BASE.Problem,
                       RDFS.label: rdflib.Literal(row["PROBABLE CAUSE"])},
                s_re: {RDFS.subClassOf: BASE.Solution,
                       RDFS.label: rdflib.Literal(row["REMEDY"]),
                       BASE.solve: s_ca},
            })

with open("prompt-extracted/part-classes.tsv") as fcls:
    with open("pdf-extracted/parts-catalog.csv") as fpc:
        with Provenance(BASE, 'parts', [fcls, fpc], OUT_DIR) as prov:
            
            def ensure_superclass_chain(cls):
                if cls in part_class_lookup:
                    parent = part_class_lookup[cls]
                    add_dict(prov.gx, {cls: {RDFS.subClassOf: parent}})
                    ensure_superclass_chain(parent)
                else:
                    add_dict(prov.gx, {cls: {RDFS.subClassOf: BASE.Part}})
        
            part_class_lookup = {}
            for row in csv.DictReader(fcls, delimiter="\t"):
                part = BASE[iri_slug(row["Part"].split("(")[0])]
                cls  = BASE[iri_slug(row["subClassOf"].split("(")[0])]
                add_dict(prov.gx, {
                    part: {RDFS.label: rdflib.Literal(row["Part"].title()),
                           RDFS.subClassOf: cls},
                    cls:  {RDFS.label: rdflib.Literal(row["subClassOf"].title())},
                })
                part_class_lookup[part] = cls
                ensure_superclass_chain(cls)
    
        
            for row in csv.DictReader(fpc):
                system  = BASE[iri_slug(row["Section"])]
                assembly= BASE[iri_slug(row["Figure"])]
                cls     = BASE[iri_slug(row["Type"])]
                label   = f'{row["Specifics"].strip()} {iri_slug(row["Type"])}'
                s       = BASE[f'partnr-{row["Part Number"]}']
    
                add_dict(prov.gx, {
                    s: {RDFS.subClassOf: cls,
                        BASE.partOf: [assembly, system],
                        BASE.partNumber: rdflib.Literal(row["Part Number"]),
                        RDFS.label: rdflib.Literal(label)},
                    assembly: {RDFS.subClassOf: BASE.Assembly,
                               RDFS.label: rdflib.Literal(row["Figure"])},
                    system: {RDFS.subClassOf: BASE.System,
                             RDFS.label: rdflib.Literal(row["Section"])},
                })
    
                if cls not in part_class_lookup:
                    add_dict(prov.gx, {cls: {
                        RDFS.subClassOf: BASE.Part,
                        RDFS.label: rdflib.Literal(iri_slug(row["Type"])),
                    }})
                else:
                    ensure_superclass_chain(cls)

with (
    open("prompt-extracted/problem-component-function.tsv") as f_pcf,
    open("prompt-extracted/functions.tsv") as f_func,
    open("prompt-extracted/subfunction.tsv") as f_sub,
    open("prompt-extracted/dependsOn.tsv") as f_dep,
):
    with Provenance(BASE, 'functions', [f_pcf, f_func, f_sub, f_dep], OUT_DIR) as prov:

        for row in csv.DictReader(f_pcf, delimiter="\t"):
            problem   = BASE[iri_slug(row["defines"].split("(")[0])]
            component = BASE[iri_slug(row["functionOf"].split("(")[0])]
            function  = BASE[iri_slug(row["Function"].split("(")[0])]
            add_dict(prov.gx, {
                function: {RDFS.subClassOf: BASE.Function,
                           RDFS.label: rdflib.Literal(row["Function"]),
                           BASE.define: problem},
                problem:  {RDFS.label: rdflib.Literal(row["defines"])},
                component:{RDFS.subClassOf: BASE.Component,
                           RDFS.label: rdflib.Literal(row["functionOf"]),
                           BASE.hasFunction: function},
            })

        for row in csv.DictReader(f_func, delimiter="\t"):
            component = BASE[iri_slug(row["Component"].split("(")[0])]
            function  = BASE[iri_slug(row["hasFunction"].split("(")[0])]
            add_dict(prov.gx, {
                function: {RDFS.label: rdflib.Literal(row["hasFunction"]),
                           RDFS.subClassOf: BASE.Function},
                component:{BASE.hasFunction: function,
                           RDFS.subClassOf: BASE.Component},
            })

        for row in csv.DictReader(f_sub, delimiter="\t"):
            func = BASE[iri_slug(row["subFunctionOf"].split("(")[0])]
            sub  = BASE[iri_slug(row["Function"].split("(")[0])]
            add_dict(prov.gx, {sub: {BASE.subFunctionOf: func}})

        for row in csv.DictReader(f_dep, delimiter="\t"):
            c1 = BASE[iri_slug(row["Component"].split("(")[0])]
            c2 = BASE[iri_slug(row["dependsOn"].split("(")[0])]
            add_dict(prov.gx, {c1: {BASE.dependsOn: c2}})


for src in ["regex", "chatgpt_4o"]:
    name = f"extractions_{src}"
    problems = f"log-extracted/problem_extractions_{src}.csv"
    actions = f"log-extracted/action_extractions_{src}.csv"
    with (
        open(problems) as f_prob,
        open(actions) as f_act,
        open("Aircraft_Annotation_DataFile.csv", encoding="utf-8-sig") as f_logs
    ):
        with Provenance(BASE, name, [f_prob, f_act, f_logs], OUT_DIR) as prov:
            # event logs index
            event_logs = {e["IDENT"]: e for e in csv.DictReader(f_logs)}
        
            def make_event(line, kind: str):
                log = event_logs[line["id"]]
                situation = BASE[f'{kind.lower()}{log["IDENT"]}']
                add_dict(prov.gx, {
                    situation: {
                        RDFS.label: rdflib.Literal(f'{kind.lower()}{log["IDENT"]}'),
                        DC.description: rdflib.Literal(log[kind.upper()]),
                    }
                })
                if line[kind]:
                    event_label = f'{line[kind].title()} {kind.title()}'
                    etype = BASE[event_label.replace(" ", "")]
                    add_dict(prov.gx, {
                        situation: {"a": etype},
                        etype: {RDFS.subClassOf: BASE[kind.title()],
                                RDFS.label: rdflib.Literal(event_label)},
                    })
        
                cyls = re.findall(r"\d", line.get("cylinders", "")) or [None]
                for c in cyls:
                    if line.get("part"):
                        part_name = re.sub(r"(S$|[^\s\w])", "", line["part"]).lower()
                        b = rdflib.BNode()
                        ptype = BASE["".join(part_name.title().split())]
                        pclass = BASE[part_name.split()[-1].title()]
                        add_dict(prov.gx, {
                            situation: {BASE.involve: b},
                            b: {"a": ptype,
                                BASE.atEngine: [rdflib.Literal(line["engine"])] if line.get("engine") else [],
                                BASE.atCylinder: [rdflib.Literal(int(c))] if c else []},
                            ptype: {RDFS.label: rdflib.Literal(part_name),
                                    RDFS.subClassOf: pclass},
                            pclass: {RDFS.subClassOf: BASE.Part},
                        })
                return situation
    
            # problems
            for row in csv.DictReader(f_prob):
                make_event(row, "problem")
            # actions
            for row in csv.DictReader(f_act):
                problem = BASE[f'problem{row["id"]}']
                action = make_event(row, "action")
                add_dict(prov.gx, {action: {BASE.dealsWith: problem}})

for src in ["regex"]:
    
    with open('part-links/part-links-regex.tsv') as fh:
        with Provenance(BASE, f'part-links-{src}', [fh], OUT_DIR) as prov:
            for raw in fh:
                part_name, part_scores = raw.split("\t", 1)
                part_uri = BASE[re.sub(r"(S$|\W)", "", part_name.title())]
                for candidate, score in json.loads(part_scores).items():
                    b = rdflib.BNode()
                    add_dict(prov.gx, {
                        part_uri: {BASE.isMaybe: b},
                        b: {BASE.linkCandidate: BASE[candidate],
                            BASE.linkScore: rdflib.Literal(score)},
                    })

    with open('part-links/part-links-regex.tsv') as fh:
        with Provenance(BASE, f'part-links-{src}-simple', [fh], OUT_DIR) as prov:
            for raw in fh:
                part_name, part_scores = raw.split("\t", 1)
                part_uri = BASE[re.sub(r"(S$|\W)", "", part_name.title())]
                candidates = collections.Counter(json.loads(part_scores))
                if candidates:
                    top, _ = candidates.most_common(1)[0]
                    add_dict(prov.gx, {part_uri: {BASE.isProbably: BASE[top]}})

