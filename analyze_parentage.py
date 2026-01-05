#!/usr/bin/env python3
"""Analyze parentage patterns for evolutionary synthesis ideas"""

from graph_repository import GraphRepository
from datetime import datetime

repo = GraphRepository()

# Find all evolutionary synthesis ideas and analyze their parentage
print("=== EVOLUTIONARY SYNTHESIS PARENTAGE ANALYSIS ===\n")

evo_ideas = []
for node_id, data in repo.graph.nodes(data=True):
    if data.get('type') == 'idea' and data.get('persona') == 'evolutionary_synthesis':
        # Count DERIVED_FROM edges
        derived_from_count = 0
        parents = []
        for _, target, edge_data in repo.graph.out_edges(node_id, data=True):
            if edge_data.get('relation') in ('DERIVED_FROM', 'EVOLVED_FROM'):
                derived_from_count += 1
                target_data = repo.graph.nodes.get(target, {})
                parents.append({
                    'id': target,
                    'persona': target_data.get('persona', 'unknown'),
                    'reason': edge_data.get('reason', '')
                })
        
        evo_ideas.append({
            'id': node_id,
            'timestamp': data.get('timestamp', 'unknown'),
            'parent_count': derived_from_count,
            'parents': parents
        })

# Sort by timestamp to see chronological pattern
evo_ideas.sort(key=lambda x: x['timestamp'])

# Summary statistics
parent_counts = {}
for idea in evo_ideas:
    pc = idea['parent_count']
    parent_counts[pc] = parent_counts.get(pc, 0) + 1

print("Parent Count Distribution:")
for count, num in sorted(parent_counts.items()):
    print(f"  {count} parents: {num} ideas")

print(f"\nTotal evolutionary_synthesis ideas: {len(evo_ideas)}\n")

# Show details for ideas with fewer than 2 parents
print("=== IDEAS WITH FEWER THAN 2 PARENTS ===\n")
for idea in evo_ideas:
    if idea['parent_count'] < 2:
        print(f"ID: {idea['id']}")
        print(f"  Timestamp: {idea['timestamp']}")
        print(f"  Parent count: {idea['parent_count']}")
        for p in idea['parents']:
            print(f"    -> {p['id'][:30]}... ({p['persona']}) - {p['reason']}")
        print()

# Show a few ideas with exactly 2 parents for comparison
print("=== IDEAS WITH EXACTLY 2 PARENTS (for comparison) ===\n")
for idea in evo_ideas[:5]:
    if idea['parent_count'] == 2:
        print(f"ID: {idea['id']}")
        print(f"  Timestamp: {idea['timestamp']}")
        for p in idea['parents']:
            print(f"    -> {p['id'][:30]}... ({p['persona']}) - {p['reason']}")
        print()
