#!/usr/bin/env python3
"""Debug script to verify relationship storage and retrieval"""

from graph_repository import GraphRepository

repo = GraphRepository()

# Find all evolutionary synthesis ideas
print("=== EVOLUTIONARY SYNTHESIS IDEAS ===")
evo_ideas = []
for node_id, data in repo.graph.nodes(data=True):
    if data.get('type') == 'idea' and data.get('persona') == 'evolutionary_synthesis':
        evo_ideas.append(node_id)
        print(f"\nID: {node_id}")
        print(f"  Content: {data.get('content', '')[:80]}...")

print(f"\nTotal evolutionary_synthesis ideas: {len(evo_ideas)}")

# Check edges for these ideas
print("\n=== CHECKING EDGES ===")
for idea_id in evo_ideas[:5]:  # Check first 5
    print(f"\n{idea_id}:")
    
    # Outgoing edges
    out_edges = list(repo.graph.out_edges(idea_id, data=True))
    print(f"  Outgoing edges ({len(out_edges)}):")
    for src, tgt, edge_data in out_edges:
        rel = edge_data.get('relation', 'UNKNOWN')
        print(f"    -> {tgt[:30]}... ({rel})")
    
    # Incoming edges
    in_edges = list(repo.graph.in_edges(idea_id, data=True))
    print(f"  Incoming edges ({len(in_edges)}):")
    for src, tgt, edge_data in in_edges:
        rel = edge_data.get('relation', 'UNKNOWN')
        print(f"    <- {src[:30]}... ({rel})")

# Test the get_idea_relationships method
print("\n=== TESTING get_idea_relationships ===")
if evo_ideas:
    test_id = evo_ideas[0]
    print(f"Testing with: {test_id}")
    relationships = repo.get_idea_relationships(test_id)
    print(f"  Parents: {relationships['parents']}")
    print(f"  Children: {relationships['children']}")
    print(f"  Semantic: {relationships['semantic']}")
