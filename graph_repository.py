
import json
import os
import networkx as nx
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import asdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Re-using these dataclasses - ideally they would be in a shared models.py
# For now, I will import them from idea_metabolism if needed, or redefine them if I want to decouple.
# To avoid circular imports, I'll assume they will be moved or I'll just use dictionaries for internal node attrs.

class GraphRepository:
    """
    Graph-based repository for ideas, problems, and domains using NetworkX.
    Nodes:
        - Problem (id, text, timestamp, embedding)
        - Idea (id, content, persona, embedding, scores)
        - Domain (id, name)
    Edges:
        - ADDRESSES (Idea -> Problem)
        - BELONGS_TO (Problem -> Domain)
        - RELATES_TO (Idea -> Idea)
        - CONTRADICTS (Idea -> Idea)
        - REQUIRES (Idea -> Idea)
    """
    
    def __init__(self, filepath: str = "idea_graph.gml"):
        self.filepath = filepath
        self.graph = nx.DiGraph()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load()

    def add_problem(self, problem_text: str, domain_name: str = "General") -> str:
        """Add a problem node and link to domain"""
        problem_id = f"prob_{hash(problem_text) % 10000000}"
        
        # Add Domain if not exists
        domain_id = f"dom_{hash(domain_name) % 10000000}"
        if not self.graph.has_node(domain_id):
            self.graph.add_node(domain_id, type="domain", name=domain_name)
            
        # Add Problem if not exists
        if not self.graph.has_node(problem_id):
            embedding = self.embedding_model.encode(problem_text).tolist()
            self.graph.add_node(
                problem_id, 
                type="problem", 
                text=problem_text, 
                timestamp=datetime.now().isoformat(),
                embedding=embedding
            )
            self.graph.add_edge(problem_id, domain_id, relation="BELONGS_TO")
            
        return problem_id

    def add_idea(self, idea_data: Any, problem_id: str):
        """Add an idea node and link to problem"""
        if hasattr(idea_data, 'to_dict'):
            idea_data = idea_data.to_dict()
        
        idea_id = idea_data['id']
        
        # Ensure embedding exists
        if 'embedding' not in idea_data or idea_data['embedding'] is None:
             idea_data['embedding'] = self.embedding_model.encode(idea_data['content']).tolist()

        if not self.graph.has_node(idea_id):
            # Store idea attributes. NetworkX supports arbitrary attrs.
            self.graph.add_node(
                idea_id,
                type="idea",
                **idea_data
            )
            self.graph.add_edge(idea_id, problem_id, relation="ADDRESSES")

    def get_top_ideas(self, n: int = 10, metric: str = "overall_interest", problem_filter: Optional[str] = None) -> List[tuple]:
         """Get top N ideas by specified metric, optionally filtered by problem"""
         # 1. Identify candidate ideas
         candidate_ids = []
         
         if problem_filter:
             # Find the problem ID for this text (fuzzy or exact)
             # For this strict filter, we might want exact alignment or lookup
             # Let's try to find the exact problem node first
             target_prob_id = None
             for node, data in self.graph.nodes(data=True):
                 if data.get('type') == 'problem' and data.get('text') == problem_filter:
                     target_prob_id = node
                     break
             
             if target_prob_id:
                 # Get ideas connected to this problem
                 candidate_ids = [n for n in self.graph.predecessors(target_prob_id) if self.graph.nodes[n].get('type') == 'idea']
             else:
                 # Problem not found, return empty
                 return []
         else:
             # All ideas
             candidate_ids = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'idea']
             
         scored_ideas = []
         for idea_id in candidate_ids:
             data = self.graph.nodes[idea_id]
             
             # Reconstruct Idea object for compatibility
             # Note: Idea constructor requires many fields, we stored them in node attrs
             # We can just return a namedtuple or object that mimics Idea, or import Idea
             # For now, let's create a simple object proxy
             class IdeaProxy:
                 def __init__(self, **kwargs):
                     self.__dict__.update(kwargs)
                     self.id = kwargs.get('id')
                     self.content = kwargs.get('content')
                     self.persona = kwargs.get('persona')
                     self.problem_context = kwargs.get('problem_context')
                     
             idea_obj = IdeaProxy(**data)
             
             evals = data.get('evaluations', [])
             if evals:
                 # Metric lookup
                 # evaluations is list of dicts
                 scores = [e.get(f"{metric}_score" if metric != "overall_interest" else "overall_interest", 0.5) 
                          for e in evals]
                 avg_score = np.mean(scores)
                 scored_ideas.append((idea_obj, avg_score))

         scored_ideas.sort(key=lambda x: x[1], reverse=True)
         return scored_ideas[:n]

    def add_evaluation(self, evaluation: Any):
        """Add an evaluation to an idea node"""
        if hasattr(evaluation, 'to_dict'):
            eval_dict = evaluation.to_dict()
        else:
            eval_dict = evaluation
            
        idea_id = eval_dict['idea_id']
        
        if self.graph.has_node(idea_id):
            node_data = self.graph.nodes[idea_id]
            if 'evaluations' not in node_data:
                node_data['evaluations'] = []
            node_data['evaluations'].append(eval_dict)

    def count_ideas(self) -> int:
        """Count total ideas in graph"""
        return len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'idea']) 

    def get_evaluations(self, idea_id: str) -> List[Any]:
        """Get evaluations for an idea"""
        if self.graph.has_node(idea_id):
            evals = self.graph.nodes[idea_id].get('evaluations', [])
            # Return as simple objects with attributes matching Evaluation class
            class EvalProxy:
                def __init__(self, **kwargs):
                   self.__dict__.update(kwargs)
            return [EvalProxy(**e) for e in evals]
        return []

    def add_relationship(self, source_idea_id: str, target_idea_id: str, relation_type: str, reason: str = ""):
        """Add semantic relationship between ideas"""
        if self.graph.has_node(source_idea_id) and self.graph.has_node(target_idea_id):
            self.graph.add_edge(source_idea_id, target_idea_id, relation=relation_type, reason=reason)

    def find_similar_problems(self, problem_text: str, threshold: float = 0.7) -> List[str]:
        """Find problem IDs semantically similar to input text"""
        query_embedding = self.embedding_model.encode(problem_text).reshape(1, -1)
        
        problem_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'problem']
        if not problem_nodes:
            return []
            
        problem_embeddings = [self.graph.nodes[n]['embedding'] for n in problem_nodes]
        
        if not problem_embeddings:
            return []
            
        similarities = cosine_similarity(query_embedding, problem_embeddings)[0]
        
        similar_problems = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                similar_problems.append((problem_nodes[idx], score))
                
        # Sort by score desc
        similar_problems.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in similar_problems]

    def get_context_ideas(self, problem_text: str) -> List[Dict[str, Any]]:
        """Retrieve all ideas linked to problems similar to the input text"""
        # 1. Find similar problems (RAG step 1)
        # Always include the exact match if it exists (threshold might miss it if logic is fuzzy)
        # We assume find_similar_problems handles the semantics.
        
        similar_problem_ids = self.find_similar_problems(problem_text, threshold=0.6) # Lower threshold for context
        
        context_ideas = []
        seen_ids = set()
        
        for pid in similar_problem_ids:
            # Get semantic neighbors (Ideas that ADDRESS this problem)
            # In DiGraph, Idea -> ADDRESSES -> Problem. So we look for predecessors of Problem.
            ideas = [n for n in self.graph.predecessors(pid) if self.graph.nodes[n].get('type') == 'idea']
            
            for idea_id in ideas:
                if idea_id not in seen_ids:
                    # Return the full node data
                    node_data = self.graph.nodes[idea_id]
                    # Make sure ID is included in the dict
                    data_copy = node_data.copy()
                    data_copy['id'] = idea_id
                    context_ideas.append(data_copy)
                    seen_ids.add(idea_id)
                    
        return context_ideas

    def save(self):
        """Save graph to GML (Good selection for node attributes)"""
        # GML doesn't support list attributes well (arrays). 
        # We need to handle serialization of embeddings manually or use Pickle/JSON-Link.
        # For this POC, let's use a custom JSON format that reconstructs the graph.
        
        data = nx.node_link_data(self.graph)
        with open(self.filepath, 'w') as f:
            json.dump(data, f)

    def load(self):
        """Load graph from JSON"""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)

    def migrate_from_json(self, json_path: str):
        """One-time migration utility"""
        if not os.path.exists(json_path):
            return
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        ideas = data.get("ideas", {})
        evaluations = data.get("evaluations", {})
        
        print(f"Migrating {len(ideas)} ideas to graph...")
        
        for idea_id, idea_dict in ideas.items():
            # 1. Create Problem Node (Deduplicated by text hash in add_problem)
            prob_text = idea_dict.get('problem_context', 'Unknown Problem')
            prob_id = self.add_problem(prob_text)
            
            # 2. Add Idea Node
            self.add_idea(idea_dict, prob_id)
            
            # 3. Add Evaluations
            if idea_id in evaluations:
                for ev in evaluations[idea_id]:
                    self.add_evaluation(ev)
            
        self.save()
        print(f"Migration complete. Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

if __name__ == "__main__":
    # Test/Migration script
    repo = GraphRepository()
    repo.migrate_from_json("idea_repository.json")
