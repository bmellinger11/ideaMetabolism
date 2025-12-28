#!/usr/bin/env python3
"""
Simple utility script to count ideas in the idea_graph.gml file
"""
from graph_repository import GraphRepository

def main():
    # Load the repository
    repo = GraphRepository(filepath="idea_graph.gml")
    
    # Count ideas
    idea_count = repo.count_ideas()
    
    # Count problems
    problem_count = len([n for n, d in repo.graph.nodes(data=True) if d.get('type') == 'problem'])
    
    # Total nodes and edges
    total_nodes = repo.graph.number_of_nodes()
    total_edges = repo.graph.number_of_edges()
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Idea Metabolism Repository Statistics")
    print(f"{'='*50}")
    print(f"Ideas:    {idea_count}")
    print(f"Problems: {problem_count}")
    print(f"Total Nodes: {total_nodes}")
    print(f"Total Edges: {total_edges}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
