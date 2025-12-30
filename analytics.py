import os
import argparse
from idea_metabolism import IdeaMetabolismSystem

def main():
    print("Loading Idea Repository...")
    # Initialize system (loads graph)
    # Use Anthropic provider just to initialize, we won't make calls
    system = IdeaMetabolismSystem(llm_provider="anthropic")
    
    print(f"Analyzing {len(system.repository.graph.nodes)} nodes...")
    
    # Find ideas with feedback
    rated_ideas = []
    
    for node_id, data in system.repository.graph.nodes(data=True):
        if data.get('type') == 'idea':
             feedback = system.repository.get_human_feedback(node_id)
             if feedback:
                 ratings = [f['rating'] for f in feedback]
                 avg_rating = sum(ratings) / len(ratings)
                 count = len(ratings)
                 
                 rated_ideas.append({
                     'id': node_id,
                     'persona': data.get('persona', 'Unknown'),
                     'content': data.get('content', '')[:60] + "...",
                     'avg_rating': avg_rating,
                     'count': count,
                     'full_content': data.get('content', '')
                 })
    
    if not rated_ideas:
        print("\nNo rated ideas found yet!")
        print("Go to the web interface and rate some ideas.")
        return

    # Sort by rating (desc), then count (desc)
    rated_ideas.sort(key=lambda x: (x['avg_rating'], x['count']), reverse=True)
    
    print("\n" + "="*80)
    print(f" COMMUNITY FAVORITES - TOP {len(rated_ideas)}")
    print("="*80)
    print(f"{'RATING':<10} | {'COUNT':<6} | {'PERSONA':<15} | {'PREVIEW'}")
    print("-" * 80)
    
    for idea in rated_ideas:
        stars = "★" * int(round(idea['avg_rating']))
        print(f"{idea['avg_rating']:.1f} {stars:<5} | {idea['count']:<6} | {idea['persona']:<15} | {idea['content']}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
