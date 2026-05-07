from dataclasses import dataclass, field

import networkx as nx
import yaml


@dataclass
class PipelineNode:
    id: str
    node_type: str  # data_source, transform, model, output
    description: str
    risk_score: float = 0.0


class PipelineDAG:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._nodes: dict[str, PipelineNode] = {}

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineDAG":
        dag = cls()
        with open(path) as f:
            config = yaml.safe_load(f)
        for n in config["nodes"]:
            dag.add_node(PipelineNode(id=n["id"], node_type=n["type"], description=n["description"]))
        for e in config["edges"]:
            dag.add_edge(e["from"], e["to"])
        return dag

    def add_node(self, node: PipelineNode) -> None:
        self._nodes[node.id] = node
        self._graph.add_node(node.id, node_type=node.node_type)

    def add_edge(self, from_id: str, to_id: str) -> None:
        self._graph.add_edge(from_id, to_id)

    def nodes(self) -> list[PipelineNode]:
        return list(self._nodes.values())

    def edges(self) -> list[tuple[str, str]]:
        return list(self._graph.edges())

    def downstream(self, node_id: str) -> list[str]:
        return list(nx.descendants(self._graph, node_id))

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._graph))

    def to_networkx(self) -> nx.DiGraph:
        return self._graph.copy()

    def update_risk_score(self, node_id: str, score: float) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].risk_score = score
