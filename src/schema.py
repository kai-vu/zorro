# Auto generated from linkml.yaml by pythongen.py version: 0.0.1
# Generation date: 2025-11-17T11:08:54
# Schema: zorro
#
# id: https://w3id.org/zorro/ontology
# description: ZORRO ontology for troubleshooting and maintenance.
# license:

import dataclasses
import re
from dataclasses import dataclass
from datetime import (
    date,
    datetime,
    time
)
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Union
)

from jsonasobj2 import (
    JsonObj,
    as_dict
)
from linkml_runtime.linkml_model.meta import (
    EnumDefinition,
    PermissibleValue,
    PvFormulaOptions
)
from linkml_runtime.utils.curienamespace import CurieNamespace
from linkml_runtime.utils.enumerations import EnumDefinitionImpl
from linkml_runtime.utils.formatutils import (
    camelcase,
    sfx,
    underscore
)
from linkml_runtime.utils.metamodelcore import (
    bnode,
    empty_dict,
    empty_list
)
from linkml_runtime.utils.slot import Slot
from linkml_runtime.utils.yamlutils import (
    YAMLRoot,
    extended_float,
    extended_int,
    extended_str
)
from rdflib import (
    Namespace,
    URIRef
)



metamodel_version = "1.7.0"
version = None

# Namespaces
IDO = CurieNamespace('ido', 'https://rds.posccaesar.org/ontology/IDO/')
IOF_CORE = CurieNamespace('iof-core', 'https://spec.industrialontologies.org/iof/ontology/core/Core/')
IOF_MAINT = CurieNamespace('iof-maint', 'https://spec.industrialontologies.org/iof/ontology/maintenance/MaintenanceReferenceOntology/')
LINKML = CurieNamespace('linkml', 'https://w3id.org/linkml/')
RDFS = CurieNamespace('rdfs', 'http://www.w3.org/2000/01/rdf-schema#')
ROMAIN = CurieNamespace('romain', 'https://github.com/HediKarray/ReferenceOntologyOfMaintenance/blob/master/ROMAIN.owl#')
SKOS = CurieNamespace('skos', 'http://www.w3.org/2004/02/skos/core#')
ZORRO = CurieNamespace('zorro', 'https://w3id.org/zorro/ontology#')
DEFAULT_ = ZORRO


# Types

# Class references



class Thing(YAMLRoot):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = RDFS["Resource"]
    class_class_curie: ClassVar[str] = "rdfs:Resource"
    class_name: ClassVar[str] = "Thing"
    class_model_uri: ClassVar[URIRef] = ZORRO.Thing


@dataclass(repr=False)
class Component(Thing):
    """
    Physical or logical component.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Component"]
    class_class_curie: ClassVar[str] = "zorro:Component"
    class_name: ClassVar[str] = "Component"
    class_model_uri: ClassVar[URIRef] = ZORRO.Component

    has_function: Optional[Union[Union[dict, "Function"], list[Union[dict, "Function"]]]] = empty_list()
    fail_via: Optional[Union[Union[dict, "Problem"], list[Union[dict, "Problem"]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if not isinstance(self.has_function, list):
            self.has_function = [self.has_function] if self.has_function is not None else []
        self.has_function = [v if isinstance(v, Function) else Function(**as_dict(v)) for v in self.has_function]

        if not isinstance(self.fail_via, list):
            self.fail_via = [self.fail_via] if self.fail_via is not None else []
        self.fail_via = [v if isinstance(v, Problem) else Problem(**as_dict(v)) for v in self.fail_via]

        super().__post_init__(**kwargs)


class System(Component):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["System"]
    class_class_curie: ClassVar[str] = "zorro:System"
    class_name: ClassVar[str] = "System"
    class_model_uri: ClassVar[URIRef] = ZORRO.System


class Assembly(Component):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Assembly"]
    class_class_curie: ClassVar[str] = "zorro:Assembly"
    class_name: ClassVar[str] = "Assembly"
    class_model_uri: ClassVar[URIRef] = ZORRO.Assembly


class Part(Component):
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Part"]
    class_class_curie: ClassVar[str] = "zorro:Part"
    class_name: ClassVar[str] = "Part"
    class_model_uri: ClassVar[URIRef] = ZORRO.Part


@dataclass(repr=False)
class Function(Thing):
    """
    Required function of a component.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Function"]
    class_class_curie: ClassVar[str] = "zorro:Function"
    class_name: ClassVar[str] = "Function"
    class_model_uri: ClassVar[URIRef] = ZORRO.Function

    depend_on: Optional[Union[Union[dict, "Function"], list[Union[dict, "Function"]]]] = empty_list()
    define: Optional[Union[dict, "Problem"]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if not isinstance(self.depend_on, list):
            self.depend_on = [self.depend_on] if self.depend_on is not None else []
        self.depend_on = [v if isinstance(v, Function) else Function(**as_dict(v)) for v in self.depend_on]

        if self.define is not None and not isinstance(self.define, Problem):
            self.define = Problem(**as_dict(self.define))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Problem(Thing):
    """
    Problem observed or inferred during operation/maintenance.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Problem"]
    class_class_curie: ClassVar[str] = "zorro:Problem"
    class_name: ClassVar[str] = "Problem"
    class_model_uri: ClassVar[URIRef] = ZORRO.Problem

    has_cause: Optional[Union[Union[dict, "Problem"], list[Union[dict, "Problem"]]]] = empty_list()
    result_in: Optional[Union[Union[dict, "Effect"], list[Union[dict, "Effect"]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if not isinstance(self.has_cause, list):
            self.has_cause = [self.has_cause] if self.has_cause is not None else []
        self.has_cause = [v if isinstance(v, Problem) else Problem(**as_dict(v)) for v in self.has_cause]

        if not isinstance(self.result_in, list):
            self.result_in = [self.result_in] if self.result_in is not None else []
        self.result_in = [v if isinstance(v, Effect) else Effect(**as_dict(v)) for v in self.result_in]

        super().__post_init__(**kwargs)


class Effect(Thing):
    """
    Consequence observed (symptom or outcome).
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Effect"]
    class_class_curie: ClassVar[str] = "zorro:Effect"
    class_name: ClassVar[str] = "Effect"
    class_model_uri: ClassVar[URIRef] = ZORRO.Effect


@dataclass(repr=False)
class Procedure(Thing):
    """
    Procedure describing a sequence of Steps.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Procedure"]
    class_class_curie: ClassVar[str] = "zorro:Procedure"
    class_name: ClassVar[str] = "Procedure"
    class_model_uri: ClassVar[URIRef] = ZORRO.Procedure

    consist_of: Optional[Union[Union[dict, "Step"], list[Union[dict, "Step"]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if not isinstance(self.consist_of, list):
            self.consist_of = [self.consist_of] if self.consist_of is not None else []
        self.consist_of = [v if isinstance(v, Step) else Step(**as_dict(v)) for v in self.consist_of]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Solution(Procedure):
    """
    Procedure intended to solve a Problem.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Solution"]
    class_class_curie: ClassVar[str] = "zorro:Solution"
    class_name: ClassVar[str] = "Solution"
    class_model_uri: ClassVar[URIRef] = ZORRO.Solution

    solve: Optional[Union[Union[dict, Problem], list[Union[dict, Problem]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if not isinstance(self.solve, list):
            self.solve = [self.solve] if self.solve is not None else []
        self.solve = [v if isinstance(v, Problem) else Problem(**as_dict(v)) for v in self.solve]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class WorkAround(Procedure):
    """
    Temporary or partial mitigation.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["WorkAround"]
    class_class_curie: ClassVar[str] = "zorro:WorkAround"
    class_name: ClassVar[str] = "WorkAround"
    class_model_uri: ClassVar[URIRef] = ZORRO.WorkAround

    address: Optional[Union[dict, Effect]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.address is not None and not isinstance(self.address, Effect):
            self.address = Effect()

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Step(Thing):
    """
    Step within a Procedure.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Step"]
    class_class_curie: ClassVar[str] = "zorro:Step"
    class_name: ClassVar[str] = "Step"
    class_model_uri: ClassVar[URIRef] = ZORRO.Step

    involve: Optional[Union[Union[dict, Component], list[Union[dict, Component]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if not isinstance(self.involve, list):
            self.involve = [self.involve] if self.involve is not None else []
        self.involve = [v if isinstance(v, Component) else Component(**as_dict(v)) for v in self.involve]

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class Action(Thing):
    """
    Generic action; added to preserve domain of deals_with.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["Action"]
    class_class_curie: ClassVar[str] = "zorro:Action"
    class_name: ClassVar[str] = "Action"
    class_model_uri: ClassVar[URIRef] = ZORRO.Action

    deals_with: Optional[Union[dict, Problem]] = None

    def __post_init__(self, *_: str, **kwargs: Any):
        if self.deals_with is not None and not isinstance(self.deals_with, Problem):
            self.deals_with = Problem(**as_dict(self.deals_with))

        super().__post_init__(**kwargs)


@dataclass(repr=False)
class ProcedureBundle(Thing):
    """
    Convenience container for instance data if needed.
    """
    _inherited_slots: ClassVar[list[str]] = []

    class_class_uri: ClassVar[URIRef] = ZORRO["ProcedureBundle"]
    class_class_curie: ClassVar[str] = "zorro:ProcedureBundle"
    class_name: ClassVar[str] = "ProcedureBundle"
    class_model_uri: ClassVar[URIRef] = ZORRO.ProcedureBundle

    procedures: Optional[Union[Union[dict, Procedure], list[Union[dict, Procedure]]]] = empty_list()

    def __post_init__(self, *_: str, **kwargs: Any):
        if not isinstance(self.procedures, list):
            self.procedures = [self.procedures] if self.procedures is not None else []
        self.procedures = [v if isinstance(v, Procedure) else Procedure(**as_dict(v)) for v in self.procedures]

        super().__post_init__(**kwargs)


# Enumerations


# Slots
class slots:
    pass

slots.deals_with = Slot(uri=ZORRO.deals_with, name="deals_with", curie=ZORRO.curie('deals_with'),
                   model_uri=ZORRO.deals_with, domain=Action, range=Optional[Union[dict, Problem]])

slots.address = Slot(uri=ZORRO.address, name="address", curie=ZORRO.curie('address'),
                   model_uri=ZORRO.address, domain=WorkAround, range=Optional[Union[dict, Effect]])

slots.consist_of = Slot(uri=ZORRO.consist_of, name="consist_of", curie=ZORRO.curie('consist_of'),
                   model_uri=ZORRO.consist_of, domain=Procedure, range=Optional[Union[Union[dict, "Step"], list[Union[dict, "Step"]]]])

slots.define = Slot(uri=ZORRO.define, name="define", curie=ZORRO.curie('define'),
                   model_uri=ZORRO.define, domain=Function, range=Optional[Union[dict, "Problem"]])

slots.depend_on = Slot(uri=ZORRO.depend_on, name="depend_on", curie=ZORRO.curie('depend_on'),
                   model_uri=ZORRO.depend_on, domain=Function, range=Optional[Union[Union[dict, "Function"], list[Union[dict, "Function"]]]])

slots.fail_via = Slot(uri=ZORRO.fail_via, name="fail_via", curie=ZORRO.curie('fail_via'),
                   model_uri=ZORRO.fail_via, domain=Component, range=Optional[Union[Union[dict, "Problem"], list[Union[dict, "Problem"]]]])

slots.has_cause = Slot(uri=ZORRO.has_cause, name="has_cause", curie=ZORRO.curie('has_cause'),
                   model_uri=ZORRO.has_cause, domain=Problem, range=Optional[Union[Union[dict, "Problem"], list[Union[dict, "Problem"]]]])

slots.has_function = Slot(uri=ZORRO.has_function, name="has_function", curie=ZORRO.curie('has_function'),
                   model_uri=ZORRO.has_function, domain=Component, range=Optional[Union[Union[dict, "Function"], list[Union[dict, "Function"]]]])

slots.involve = Slot(uri=ZORRO.involve, name="involve", curie=ZORRO.curie('involve'),
                   model_uri=ZORRO.involve, domain=Step, range=Optional[Union[Union[dict, Component], list[Union[dict, Component]]]])

slots.result_in = Slot(uri=ZORRO.result_in, name="result_in", curie=ZORRO.curie('result_in'),
                   model_uri=ZORRO.result_in, domain=Problem, range=Optional[Union[Union[dict, "Effect"], list[Union[dict, "Effect"]]]])

slots.solve = Slot(uri=ZORRO.solve, name="solve", curie=ZORRO.curie('solve'),
                   model_uri=ZORRO.solve, domain=Solution, range=Optional[Union[Union[dict, Problem], list[Union[dict, Problem]]]])

slots.procedureBundle__procedures = Slot(uri=ZORRO.procedures, name="procedureBundle__procedures", curie=ZORRO.curie('procedures'),
                   model_uri=ZORRO.procedureBundle__procedures, domain=None, range=Optional[Union[Union[dict, Procedure], list[Union[dict, Procedure]]]])

