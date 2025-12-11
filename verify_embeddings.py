
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def test_embeddings():
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    words = ["cat", "kitten", "banana", "fruit"]
    print(f"Generating embeddings for: {words}")
    
    embeddings = model.encode(words)
    
    # Calculate similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Check cat vs kitten (should be high)
    cat_kitten = sim_matrix[0][1]
    print(f"Similarity 'cat' vs 'kitten': {cat_kitten:.4f}")
    
    # Check cat vs banana (should be low)
    cat_banana = sim_matrix[0][2]
    print(f"Similarity 'cat' vs 'banana': {cat_banana:.4f}")
    
    # Check banana vs fruit (should be high)
    banana_fruit = sim_matrix[2][3]
    print(f"Similarity 'banana' vs 'fruit': {banana_fruit:.4f}")
    
    if cat_kitten > cat_banana and banana_fruit > cat_banana:
        print("\nSUCCESS: Semantic similarity verified!")
    else:
        print("\nFAILURE: Semantic relationships not preserved.")

if __name__ == "__main__":
    test_embeddings()
