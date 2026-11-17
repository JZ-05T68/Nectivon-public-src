"""Course-scoped projections retain shared identities and real relation semantics."""

from __future__ import annotations

from dataclasses import replace

from src.knowledge_graph_service import KnowledgeGraph


def subject_graph(graph: KnowledgeGraph, subject: str) -> KnowledgeGraph:
    """Select one subject plus its source anchors, without inventing a tree."""
    identifiers = {identity for identity, node in graph.nodes.items()
                   if (not node.get("tags") and subject == "未归类"
                       and node["entity_kind"] not in {"page", "document"})
                   or any(tag.split("/", 1)[0] == subject for tag in node.get("tags", []))}
    # Source/document anchors may have no subject of their own. Include only
    # actual source references, never other knowledge through spatial adjacency.
    while True:
        sources = {edge[end] for edge in graph.links for end in ("source", "target")
                   if edge["source"] in identifiers or edge["target"] in identifiers
                   if graph.nodes[edge[end]]["entity_kind"] in {"page", "document"}}
        if sources.issubset(identifiers):
            break
        identifiers.update(sources)
    nodes = {key: node for key, node in graph.nodes.items() if key in identifiers}
    links = [edge for edge in graph.links
             if edge["source"] in identifiers and edge["target"] in identifiers]
    count = sum(node["entity_kind"] not in {"page", "document"} for node in nodes.values())
    return replace(graph, nodes=nodes, links=links, total=count, shown=count)


def search_graph(graph: KnowledgeGraph, query: str) -> KnowledgeGraph:
    """Search a preview locally while preserving the same candidate identities."""
    query = query.strip().casefold()
    if not query:
        return graph
    nodes = {key: node for key, node in graph.nodes.items()
             if query in (node.get("title", "") + "\n" + node.get("summary", "")).casefold()}
    links = [edge for edge in graph.links if edge["source"] in nodes and edge["target"] in nodes]
    return replace(graph, nodes=nodes, links=links, total=len(nodes), shown=len(nodes))
