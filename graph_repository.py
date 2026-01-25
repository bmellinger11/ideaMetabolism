
import json
import os
import networkx as nx
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import asdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from filelock import FileLock

logger = logging.getLogger(__name__)


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
        self.lock_path = f"{filepath}.lock"
        self.lock = FileLock(self.lock_path)
        self.graph = nx.DiGraph()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load()

    def add_problem(self, problem_text: str, domain_name: str = "General") -> str:
        """Add a problem node and link to domain"""
        problem_id = f"prob_{hash(problem_text) % 10000000}"
        
        with self.lock:
            self.load()
            
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

            self.save()
            
        return problem_id

    def add_idea(self, idea_data: Any, problem_id: str):
        """Add an idea node and link to problem"""
        if hasattr(idea_data, 'to_dict'):
            idea_data = idea_data.to_dict()
        
        idea_id = idea_data['id']
        
        # Ensure embedding exists (do before lock to save time)
        if 'embedding' not in idea_data or idea_data['embedding'] is None:
            idea_data['embedding'] = self.embedding_model.encode(idea_data['content']).tolist()

        with self.lock:
            self.load()
            
            if not self.graph.has_node(idea_id):
                # Store idea attributes. NetworkX supports arbitrary attrs.
                self.graph.add_node(
                    idea_id,
                    type="idea",
                    **idea_data
                )
                self.graph.add_edge(idea_id, problem_id, relation="ADDRESSES")
            
            self.save()

    def get_top_ideas(
            self,
            n: int = 10,
            metric: str = "overall_interest",
            problem_filter: Optional[str] = None
    ) -> List[tuple]:
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
                candidate_ids = [
                    n for n in self.graph.predecessors(target_prob_id)
                    if self.graph.nodes[n].get('type') == 'idea'
                ]
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
            class IdeaProxy:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                    self.id = kwargs.get('id')
                    self.content = kwargs.get('content')
                    self.persona = kwargs.get('persona')
                    self.problem_context = kwargs.get('problem_context')
                    self.timestamp = kwargs.get('timestamp')
                    
            idea_obj = IdeaProxy(**data)
            if not getattr(idea_obj, 'id', None):
                idea_obj.id = idea_id
            
            scored_ideas.append((idea_obj, self.get_score_breakdown(idea_id, metric)['combined_score']))

        scored_ideas.sort(key=lambda x: x[1], reverse=True)
        return scored_ideas[:n]

    def add_evaluation(self, evaluation: Any):
        """Add an evaluation to an idea node"""
        if hasattr(evaluation, 'to_dict'):
            eval_dict = evaluation.to_dict()
        else:
            eval_dict = evaluation
            
        idea_id = eval_dict['idea_id']
        
        with self.lock:
            self.load()
            if self.graph.has_node(idea_id):
                node_data = self.graph.nodes[idea_id]
                if 'evaluations' not in node_data:
                    node_data['evaluations'] = []
                node_data['evaluations'].append(eval_dict)
            self.save()

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
    
    def add_human_feedback(self, idea_id: str, rating: int, comment: str, timestamp: str):
        """Add human feedback to an idea node"""
        with self.lock:
            self.load()
            
            if idea_id not in self.graph.nodes:
                # Should we raise? If another process deleted it?
                # For now, just log/return or raise as before
                # But if we assume it might be missing due to sync issues, reloading fixes that.
                # If it's still missing, it's missing.
                pass 
            
            if idea_id in self.graph.nodes:
                # Get or create human_feedback list
                if 'human_feedback' not in self.graph.nodes[idea_id]:
                    self.graph.nodes[idea_id]['human_feedback'] = []
                
                # Add feedback entry
                feedback = {
                    'rating': rating,
                    'comment': comment,
                    'timestamp': timestamp,
                    'source': 'human'
                }
                
                self.graph.nodes[idea_id]['human_feedback'].append(feedback)
                
                # Save to disk
                self.save()
                
        logger.info(f"Added human feedback to {idea_id}: {rating} stars")
    
    def get_human_feedback(self, idea_id: str) -> List[dict]:
        """Get all human feedback for an idea"""
        if self.graph.has_node(idea_id):
            return self.graph.nodes[idea_id].get('human_feedback', [])
        return []

    def get_score_breakdown(self, idea_id: str, metric: str = "overall_interest") -> Dict[str, Any]:
        """Calculate detailed breakdown of AI vs Human scores"""
        data = self.graph.nodes.get(idea_id, {})
        if not data:
            return {}

        # 1. AI Score
        evals = data.get('evaluations', [])
        ai_score = 0.5
        
        # Calculate dynamic novelty
        dynamic_novelty = self.compute_current_novelty(idea_id)
        
        if metric == "novelty":
            return {"combined_score": dynamic_novelty, "ai_score": dynamic_novelty, "boost_reason": ""}

        if evals:
            # We use the dynamic novelty instead of the stored one
            # The AI score typically averages multiple dimensions. 
            # We'll treat novelty as one equal weighted component if we are just averaging.
            # OR we can assume the user wants `novelty` metric specifically.
            
            # Standard weighted average of components? 
            # The current impl was just averaging generic scores.
            # Let's reconstruct the average but swap in dynamic novelty.
            
            # Extract other scores from latest eval (assuming single authoritative eval for now)
            latest_eval = evals[-1]
            
            # Components from Evaluation class
            feasibility = latest_eval.get('feasibility_score', 0.5)
            surprise = latest_eval.get('surprise_score', 0.5)
            # coherence = latest_eval.get('coherence_score', 0.5) 
            # generativity = latest_eval.get('generativity_score', 0.5)
            interest = latest_eval.get('overall_interest', 0.5)

            if metric == "overall_interest":
                # If specifically asking for overall interest, we might want to re-calculate it 
                # as a function of the new novelty?
                # The original "overall_interest" from the LLM is a "gut check". 
                # But if novelty drops, interest SHOULD drop.
                # Simple heuristic: re-weight interest based on novelty change?
                # For simplicity/robustness: Let's average (Novelty + Feasibility + Surprise + Interest) / 4
                # This ensures novelty directly impacts the ranking.
                scores = [dynamic_novelty, feasibility, surprise, interest]
                ai_score = float(np.mean(scores))
            else:
                # Fallback for specific metric requests other than novelty/interest
                score_key = f"{metric}_score"
                ai_score = latest_eval.get(score_key, 0.5)

        
        # 2. Human Feedback
        feedback = data.get('human_feedback', [])
        avg_rating = 0.0
        feedback_count = len(feedback)
        boost_multiplier = 1.0
        boost_reason = ""

        if feedback_count > 0:
            ratings = [f['rating'] for f in feedback]
            avg_rating = sum(ratings) / feedback_count
            
            if avg_rating >= 4.5:
                boost_multiplier = 1.50
                boost_reason = f"High User Rating ({avg_rating:.1f})"
            elif avg_rating >= 4.0:
                boost_multiplier = 1.25
                boost_reason = f"Positive User Rating ({avg_rating:.1f})"
            elif avg_rating >= 3.0:
                boost_multiplier = 1.10
            elif avg_rating >= 2.0:
                boost_multiplier = 0.80
                boost_reason = f"Low User Rating ({avg_rating:.1f})"
            else:
                boost_multiplier = 0.50
                boost_reason = f"Negative User Rating ({avg_rating:.1f})"
        
        combined_score = ai_score * boost_multiplier

        return {
            "ai_score": ai_score,
            "avg_rating": avg_rating,
            "feedback_count": feedback_count,
            "boost_multiplier": boost_multiplier,
            "boost_reason": boost_reason,
            "combined_score": combined_score
        }

    def add_relationship(self, source_idea_id: str, target_idea_id: str, relation_type: str, reason: str = ""):
        """Add semantic relationship between ideas"""
        with self.lock:
            self.load()
            if self.graph.has_node(source_idea_id) and self.graph.has_node(target_idea_id):
                self.graph.add_edge(source_idea_id, target_idea_id, relation=relation_type, reason=reason)
            self.save()

    def get_idea_relationships(self, idea_id: str) -> Dict[str, List[Dict]]:
        """Get all relationships for an idea, grouped by type.
        
        Returns:
            {
                "parents": [{"id": ..., "persona": ..., "content_preview": ..., "reason": ...}],
                "children": [...],
                "semantic": [{"id": ..., "type": ..., "reason": ...}]
            }
        """
        if not self.graph.has_node(idea_id):
            return {"parents": [], "children": [], "semantic": []}
        
        parents = []
        children = []
        semantic = []
        
        # Outgoing edges (current idea is source)
        for _, target, edge_data in self.graph.out_edges(idea_id, data=True):
            relation = edge_data.get("relation", "")
            reason = edge_data.get("reason", "")
            
            # Skip non-idea nodes (e.g., ADDRESSES -> Problem)
            target_data = self.graph.nodes.get(target, {})
            if target_data.get("type") != "idea":
                continue
            
            if relation in ("EVOLVED_FROM", "DERIVED_FROM"):
                parents.append({
                    "id": target,
                    "persona": target_data.get("persona", "Unknown"),
                    "content_preview": target_data.get("content", "")[:100] + "...",
                    "reason": reason
                })
            else:
                semantic.append({
                    "id": target,
                    "persona": target_data.get("persona", "Unknown"),
                    "content_preview": target_data.get("content", "")[:100] + "...",
                    "type": relation,
                    "reason": reason
                })
        
        # Incoming edges (current idea is target)
        for source, _, edge_data in self.graph.in_edges(idea_id, data=True):
            relation = edge_data.get("relation", "")
            reason = edge_data.get("reason", "")
            
            source_data = self.graph.nodes.get(source, {})
            if source_data.get("type") != "idea":
                continue
            
            if relation in ("EVOLVED_FROM", "DERIVED_FROM"):
                # If something evolved FROM this idea, this idea is a parent
                children.append({
                    "id": source,
                    "persona": source_data.get("persona", "Unknown"),
                    "content_preview": source_data.get("content", "")[:100] + "...",
                    "reason": reason
                })
            else:
                semantic.append({
                    "id": source,
                    "persona": source_data.get("persona", "Unknown"),
                    "content_preview": source_data.get("content", "")[:100] + "...",
                    "type": f"{relation} (incoming)",
                    "reason": reason
                })
        
        return {"parents": parents, "children": children, "semantic": semantic}

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
        # Step 1: Try exact text match first
        exact_match_id = None
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'problem' and data.get('text') == problem_text:
                exact_match_id = node
                break
        
        # Step 2: Semantic search with lower threshold for fuzzy matching
        similar_problem_ids = self.find_similar_problems(problem_text, threshold=0.5)
        
        # Step 3: Ensure exact match is included if found (even if below semantic threshold)
        if exact_match_id and exact_match_id not in similar_problem_ids:
            similar_problem_ids.insert(0, exact_match_id)
        
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
                    
                    breakdown = self.get_score_breakdown(idea_id)
                    data_copy['score'] = breakdown['combined_score']
                    if breakdown['boost_reason']:
                        data_copy['boost_reason'] = breakdown['boost_reason']
                            
                    context_ideas.append(data_copy)
                    seen_ids.add(idea_id)
        
        # Sort by boosted score desc
        context_ideas.sort(key=lambda x: x.get('score', 0), reverse=True)
                    
        return context_ideas

    def save(self):
        """Save graph to GML (Good selection for node attributes)"""
        # GML doesn't support list attributes well (arrays). 
        # We need to handle serialization of embeddings manually or use Pickle/JSON-Link.
        # For this POC, let's use a custom JSON format that reconstructs the graph.
        
        data = nx.node_link_data(self.graph, edges="links")
        with open(self.filepath, 'w') as f:
            json.dump(data, f)

    def load(self):
        """Load graph from JSON"""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data, edges="links")

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
        print(
            f"Migration complete. "
            f"Graph has {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges."
        )


    def compute_current_novelty(self, idea_id: str) -> float:
        """Compute novelty as distance from other ideas in the SAME problem context"""
        if not self.graph.has_node(idea_id):
            return 0.0
            
        target_embedding = self.graph.nodes[idea_id].get('embedding')
        if target_embedding is None:
            return 0.0
            
        # 1. Find the problem(s) this idea addresses
        # In DiGraph: Idea -> ADDRESSES -> Problem
        # So Problem is a successor of Idea
        problem_ids = [
            n for n in self.graph.successors(idea_id) 
            if self.graph.nodes[n].get('type') == 'problem'
        ]
        
        if not problem_ids:
            # Fallback: compare against all if no problem context found (orphan idea)
            # Or return default? Let's fallback for safety.
            candidates = [n for n in self.graph.nodes if self.graph.nodes[n].get('type') == 'idea' and n != idea_id]
        else:
            # 2. Find all ideas that address these problems (Siblings)
            # Idea -> Problem. So we look for Predecessors of the Problem nodes.
            candidates = set()
            for pid in problem_ids:
                siblings = [
                    n for n in self.graph.predecessors(pid) 
                    if self.graph.nodes[n].get('type') == 'idea' and n != idea_id
                ]
                candidates.update(siblings)
            candidates = list(candidates)
            
        if not candidates:
            return 0.8  # Unique in its problem space

        # Get embeddings
        other_embeddings = []
        for n in candidates:
            emb = self.graph.nodes[n].get('embedding')
            if emb is not None:
                other_embeddings.append(emb)
        
        if not other_embeddings:
            return 0.8

        # Compute cosine similarity
        similarities = cosine_similarity([target_embedding], other_embeddings)[0]
        max_similarity = np.max(similarities)
        
        return 1.0 - max_similarity


if __name__ == "__main__":
    # Test/Migration script
    repo = GraphRepository()
    repo.migrate_from_json("idea_repository.json")
