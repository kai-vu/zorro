import csv, re, json, collections, logging
from pathlib import Path

import rdflib
from rdflib import RDFS
from rdflib.namespace import DC
import tqdm

from make_prov import rule, build  # new module

# ---------- helpers ----------
def iri_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", " ".join(s.split()[:10]).title())

def add_dict(g, mapping: dict):
    for s, po in mapping.items():
        for p, o in po.items():
            pred = RDF.type if p == "a" else p
            objs = o if isinstance(o, (list, tuple, set)) else [o]
            for obj in objs:
                g.add((s, pred, obj))

# ---------- namespaces ----------
BASE = rdflib.Namespace("https://w3id.org/zorro#")
OUT_DIR = "generated-trig"



# troubleshooting


@rule(
    target=f"{OUT_DIR}/troubleshooting.trig",
    deps=["pdf-extracted/troubleshooting.csv"],
    base_iri=str(BASE),
    name="troubleshooting",
    prov_dir=OUT_DIR,
)
def build_troubleshooting():
    g = rdflib.Graph()
    g.bind("", BASE)

    with open("pdf-extracted/troubleshooting.csv") as fh:
        for row in csv.DictReader(fh):
            s_tr = BASE[iri_slug(row["TROUBLE"])]
            s_ca = BASE[iri_slug(row["PROBABLE CAUSE"])]
            s_re = BASE[iri_slug(row["REMEDY"])]
            add_dict(g, {
                s_tr: {RDFS.subClassOf: BASE.Problem,
                       RDFS.label: rdflib.Literal(row["TROUBLE"]),
                       BASE.hasCause: s_ca},
                s_ca: {RDFS.subClassOf: BASE.Problem,
                       RDFS.label: rdflib.Literal(row["PROBABLE CAUSE"])},
                s_re: {RDFS.subClassOf: BASE.Solution,
                       RDFS.label: rdflib.Literal(row["REMEDY"]),
                       BASE.solve: s_ca},
            })

    return g



# parts


@rule(
    target=f"{OUT_DIR}/parts.trig",
    deps=[
        "prompt-extracted/part-classes.tsv",
        "pdf-extracted/parts-catalog.csv",
    ],
    base_iri=str(BASE),
    name="parts",
    prov_dir=OUT_DIR,
)
def build_parts():
    g = rdflib.Graph()
    g.bind("", BASE)

    def ensure_superclass_chain(cls):
        if cls in part_class_lookup:
            parent = part_class_lookup[cls]
            add_dict(g, {cls: {RDFS.subClassOf: parent}})
            ensure_superclass_chain(parent)
        else:
            add_dict(g, {cls: {RDFS.subClassOf: BASE.Part}})

    with (
        open("prompt-extracted/part-classes.tsv") as fcls,
        open("pdf-extracted/parts-catalog.csv") as fpc,
    ):
        part_class_lookup = {}

        # from prompt-extracted classes
        for row in csv.DictReader(fcls, delimiter="\t"):
            part = BASE[iri_slug(row["Part"].split("(")[0])]
            cls  = BASE[iri_slug(row["subClassOf"].split("(")[0])]
            add_dict(g, {
                part: {RDFS.label: rdflib.Literal(row["Part"].title()),
                       RDFS.subClassOf: cls},
                cls:  {RDFS.label: rdflib.Literal(row["subClassOf"].title())},
            })
            part_class_lookup[part] = cls
            ensure_superclass_chain(cls)

        # from parts-catalog
        for row in csv.DictReader(fpc):
            system   = BASE[iri_slug(row["Section"])]
            assembly = BASE[iri_slug(row["Figure"])]
            cls      = BASE[iri_slug(row["Type"])]
            label    = f'{row["Specifics"].strip()} {iri_slug(row["Type"])}'
            s        = BASE[f'partnr-{row["Part Number"]}']

            add_dict(g, {
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
                add_dict(g, {cls: {
                    RDFS.subClassOf: BASE.Part,
                    RDFS.label: rdflib.Literal(iri_slug(row["Type"])),
                }})
            else:
                ensure_superclass_chain(cls)

    return g



# functions


@rule(
    target=f"{OUT_DIR}/functions.trig",
    deps=[
        "prompt-extracted/problem-component-function.tsv",
        "prompt-extracted/functions.tsv",
        "prompt-extracted/subfunction.tsv",
        "prompt-extracted/dependsOn.tsv",
    ],
    base_iri=str(BASE),
    name="functions",
    prov_dir=OUT_DIR,
)
def build_functions():
    g = rdflib.Graph()
    g.bind("", BASE)

    with (
        open("prompt-extracted/problem-component-function.tsv") as f_pcf,
        open("prompt-extracted/functions.tsv") as f_func,
        open("prompt-extracted/subfunction.tsv") as f_sub,
        open("prompt-extracted/dependsOn.tsv") as f_dep,
    ):
        for row in csv.DictReader(f_pcf, delimiter="\t"):
            problem   = BASE[iri_slug(row["defines"].split("(")[0])]
            component = BASE[iri_slug(row["functionOf"].split("(")[0])]
            function  = BASE[iri_slug(row["Function"].split("(")[0])]
            add_dict(g, {
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
            add_dict(g, {
                function: {RDFS.label: rdflib.Literal(row["hasFunction"]),
                           RDFS.subClassOf: BASE.Function},
                component:{BASE.hasFunction: function,
                           RDFS.subClassOf: BASE.Component},
            })

        for row in csv.DictReader(f_sub, delimiter="\t"):
            func = BASE[iri_slug(row["subFunctionOf"].split("(")[0])]
            sub  = BASE[iri_slug(row["Function"].split("(")[0])]
            add_dict(g, {sub: {BASE.subFunctionOf: func}})

        for row in csv.DictReader(f_dep, delimiter="\t"):
            c1 = BASE[iri_slug(row["Component"].split("(")[0])]
            c2 = BASE[iri_slug(row["dependsOn"].split("(")[0])]
            add_dict(g, {c1: {BASE.dependsOn: c2}})

    return g



# extractions (per src: regex, chatgpt_4o)


def _register_extractions_rule(src: str):
    problems = f"log-extracted/problem_extractions_{src}.csv"
    actions  = f"log-extracted/action_extractions_{src}.csv"
    target   = f"{OUT_DIR}/extractions_{src}.trig"
    name     = f"extractions_{src}"

    @rule(
        target=target,
        deps=[problems, actions, "Aircraft_Annotation_DataFile.csv"],
        base_iri=str(BASE),
        name=name,
        prov_dir=OUT_DIR,
    )
    def build_extractions():
        g = rdflib.Graph()
        g.bind("", BASE)
        g.bind("dc", DC)

        with (
            open(problems) as f_prob,
            open(actions) as f_act,
            open("Aircraft_Annotation_DataFile.csv", encoding="utf-8-sig") as f_logs,
        ):
            # event logs index
            event_logs = {e["IDENT"]: e for e in csv.DictReader(f_logs)}

            def make_event(line, kind: str):
                log = event_logs[line["id"]]
                situation = BASE[f'{kind.lower()}{log["IDENT"]}']
                add_dict(g, {
                    situation: {
                        RDFS.label: rdflib.Literal(f'{kind.lower()}{log["IDENT"]}'),
                        DC.description: rdflib.Literal(log[kind.upper()]),
                    }
                })
                if line[kind]:
                    event_label = f'{line[kind].title()} {kind.title()}'
                    etype = BASE[event_label.replace(" ", "")]
                    add_dict(g, {
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
                        add_dict(g, {
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
            for row in tqdm.tqdm(list(csv.DictReader(f_prob))):
                make_event(row, "problem")
            # actions
            for row in tqdm.tqdm(list(csv.DictReader(f_act))):
                problem = BASE[f'problem{row["id"]}']
                action = make_event(row, "action")
                add_dict(g, {action: {BASE.dealsWith: problem}})

        return g

    return build_extractions


for src in ["regex", "chatgpt_4o"]:
    _register_extractions_rule(src)



# part-links (regex)


def _register_part_links_rules(src: str):
    in_path = "part-links/part-links-regex.tsv"

    @rule(
        target=f"{OUT_DIR}/part-links-{src}.trig",
        deps=[in_path],
        base_iri=str(BASE),
        name=f"part-links-{src}",
        prov_dir=OUT_DIR,
    )
    def build_part_links_full():
        g = rdflib.Graph()
        g.bind("", BASE)

        with open(in_path) as fh:
            for raw in fh:
                part_name, part_scores = raw.split("\t", 1)
                part_uri = BASE[re.sub(r"(S$|\W)", "", part_name.title())]
                for candidate, score in json.loads(part_scores).items():
                    b = rdflib.BNode()
                    add_dict(g, {
                        part_uri: {BASE.isMaybe: b},
                        b: {BASE.linkCandidate: BASE[candidate],
                            BASE.linkScore: rdflib.Literal(score)},
                    })

        return g

    @rule(
        target=f"{OUT_DIR}/part-links-{src}-simple.trig",
        deps=[in_path],
        base_iri=str(BASE),
        name=f"part-links-{src}-simple",
        prov_dir=OUT_DIR,
    )
    def build_part_links_simple():
        g = rdflib.Graph()
        g.bind("", BASE)

        with open(in_path) as fh:
            for raw in fh:
                part_name, part_scores = raw.split("\t", 1)
                part_uri = BASE[re.sub(r"(S$|\W)", "", part_name.title())]
                candidates = collections.Counter(json.loads(part_scores))
                if candidates:
                    top, _ = candidates.most_common(1)[0]
                    add_dict(g, {part_uri: {BASE.isProbably: BASE[top]}})

        return g

    return build_part_links_full, build_part_links_simple


for src in ["regex"]:
    _register_part_links_rules(src)



if __name__ == "__main__":
    # Call whichever targets you want to build
    build(f"{OUT_DIR}/troubleshooting.trig")
    build(f"{OUT_DIR}/parts.trig")
    build(f"{OUT_DIR}/functions.trig")

    for src in ["regex", "chatgpt_4o"]:
        build(f"{OUT_DIR}/extractions_{src}.trig")

    for src in ["regex"]:
        build(f"{OUT_DIR}/part-links-{src}.trig")
        build(f"{OUT_DIR}/part-links-{src}-simple.trig")
