
from idea_metabolism import IdeaMetabolismSystem
import sys

def verify_feedback(idea_id):
    print(f"Checking feedback for: {idea_id}")
    # Initialize system (loads graph)
    system = IdeaMetabolismSystem(llm_provider="anthropic")
    
    if idea_id not in system.repository.graph.nodes:
        print(f"Error: Idea {idea_id} not found in repository.")
        return

    feedback = system.repository.get_human_feedback(idea_id)
    
    if not feedback:
        print("No feedback found for this idea.")
    else:
        print(f"\nFound {len(feedback)} feedback entries:")
        for i, f in enumerate(feedback, 1):
            print(f"\nEntry #{i}:")
            print(f"  Rating: {f['rating']} stars")
            print(f"  Comment: {f.get('comment', '(no comment)')}")
            print(f"  Timestamp: {f['timestamp']}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_feedback.py <idea_id>")
        sys.exit(1)
    verify_feedback(sys.argv[1])
