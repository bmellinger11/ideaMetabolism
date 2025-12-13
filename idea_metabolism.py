"""
Idea Metabolism System: Stage 1 (Generation) & Stage 2 (Triage)

A multi-agent system for generating and evaluating novel ideas using
diverse personas and building a queryable knowledge repository.

Usage:
    python idea_metabolism.py --problem "your problem statement"
    python idea_metabolism.py --problem "your problem statement" --repo-only 5
    
Requirements:
    pip install anthropic openai google-generativeai numpy scikit-learn
"""

import os
import json
import time
import argparse
import sys
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Literal
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from graph_repository import GraphRepository


# Replace with your preferred LLM
# Load environment variables
load_dotenv()  # Load variables from .env into the environment
if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")


# LLM Client Setup
class LLMClient:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self, provider: Literal["anthropic", "openai", "gemini"] = "anthropic"):
        self.provider = provider
        
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            # Initialize local embedding model for Anthropic
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.client = genai
            # Initialize local embedding model for Gemini
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text using configured provider"""
        
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4",
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
        elif self.provider == "gemini":
            model = self.client.GenerativeModel('gemini-pro')
            response = model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
            return response.text
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text"""
        
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        else:
            # Use local sentence-transformers model
            return self.embedding_model.encode(text).tolist()


@dataclass
class Idea:
    """Represents a generated idea"""
    id: str
    content: str
    persona: str
    temperature: float
    timestamp: str
    problem_context: str
    embedding: Optional[List[float]] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Evaluation:
    """Represents an evaluation of an idea"""
    idea_id: str
    evaluator: str  # "agent" or "human"
    novelty_score: float  # 0-1
    feasibility_score: float  # 0-1
    surprise_score: float  # 0-1
    coherence_score: float  # 0-1
    generativity_score: float  # 0-1
    overall_interest: float  # 0-1
    reasoning: str
    timestamp: str
    
    def to_dict(self):
        return asdict(self)


class PersonaAgent:
    """Agent with specific persona characteristics"""
    
    PERSONAS = {
        "convergent": {
            "name": "Convergent Thinker",
            "temperature": 0.1,
            "prompt_style": """You are a careful, methodical thinker who builds on established knowledge.
Generate ideas that are incremental improvements, well-grounded in existing approaches.
Focus on practical, implementable solutions with clear paths forward."""
        },
        "alternative": {
            "name": "Alternative Thinker", 
            "temperature": 0.3,
            "prompt_style": """You are a perspective-shifting thinker who sees problems from unusual angles.
Generate ideas that reframe the problem, draw analogies from other domains, or challenge assumptions.
Focus on questioning what's taken for granted."""
        },
        "divergent": {
            "name": "Divergent Thinker",
            "temperature": 0.7,
            "prompt_style": """You are a radically creative thinker who explores the boundaries of possibility.
Generate ideas that break conventions, violate apparent constraints, or seem impossible at first.
Focus on maximally different approaches, even if they seem impractical."""
        }
    }
    
    def __init__(self, persona_type: str, llm_client: LLMClient):
        self.persona_type = persona_type
        self.config = self.PERSONAS[persona_type]
        self.llm = llm_client
    
    def generate_ideas(self, problem: str, num_ideas: int = 5) -> List[Idea]:
        """Generate multiple ideas for a problem"""
        
        prompt = f"""{self.config['prompt_style']}

PROBLEM: {problem}

Generate {num_ideas} distinct ideas to address this problem. For each idea:
1. State the core concept clearly
2. Explain why it might work
3. Identify key assumptions or prerequisites

Format as JSON array:
[
  {{"idea": "...", "rationale": "...", "assumptions": "..."}},
  ...
]"""

        response = self.llm.generate(
            prompt, 
            temperature=self.config['temperature'],
            max_tokens=3000
        )
        
        # Parse ideas from response
        ideas = []
        try:
            # Try to extract JSON
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                ideas_data = json.loads(response[start:end])
            else:
                # Fallback: treat as single idea
                ideas_data = [{"idea": response, "rationale": "", "assumptions": ""}]
        except json.JSONDecodeError:
            # Fallback: treat entire response as single idea
            ideas_data = [{"idea": response, "rationale": "", "assumptions": ""}]
        
        # Create Idea objects
        result = []
        for i, idea_data in enumerate(ideas_data[:num_ideas]):
            idea_text = f"{idea_data.get('idea', '')}\n\nRationale: {idea_data.get('rationale', '')}\nAssumptions: {idea_data.get('assumptions', '')}"
            
            idea = Idea(
                id=f"{self.persona_type}_{int(time.time())}_{i}",
                content=idea_text,
                persona=self.persona_type,
                temperature=self.config['temperature'],
                timestamp=datetime.now().isoformat(),
                problem_context=problem,
                embedding=None  # Will be computed later
            )
            result.append(idea)
        
        return result


class IdeaEvaluator:
    """Evaluates ideas along multiple dimensions"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def evaluate_idea(self, idea: Idea, existing_ideas: List[Idea]) -> Evaluation:
        """Evaluate an idea with multi-dimensional scoring"""
        
        # Compute novelty based on embedding distance
        novelty = self._compute_novelty(idea, existing_ideas)
        
        # Use LLM to evaluate other dimensions
        prompt = f"""Evaluate this idea along multiple dimensions:

IDEA: {idea.content}

PROBLEM CONTEXT: {idea.problem_context}

Rate each dimension from 0.0 (lowest) to 1.0 (highest):

1. FEASIBILITY: Can we actually test/implement this? Do we know how to evaluate it?
2. SURPRISE: How unexpected or counter-intuitive is this approach?
3. COHERENCE: Is it internally consistent and well-reasoned?
4. GENERATIVITY: Does it suggest further ideas or open new directions?

Also provide:
5. OVERALL INTEREST: Your gut assessment of how interesting this is
6. REASONING: Brief explanation of your scores (2-3 sentences)

Respond in JSON format:
{{
  "feasibility": 0.0-1.0,
  "surprise": 0.0-1.0,
  "coherence": 0.0-1.0,
  "generativity": 0.0-1.0,
  "overall_interest": 0.0-1.0,
  "reasoning": "..."
}}"""

        response = self.llm.generate(prompt, temperature=0.3)
        
        # Parse evaluation
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            eval_data = json.loads(response[start:end])
        except json.JSONDecodeError:
            # Fallback scores
            eval_data = {
                "feasibility": 0.5,
                "surprise": 0.5,
                "coherence": 0.5,
                "generativity": 0.5,
                "overall_interest": 0.5,
                "reasoning": "Could not parse evaluation"
            }
        
        return Evaluation(
            idea_id=idea.id,
            evaluator="agent",
            novelty_score=novelty,
            feasibility_score=eval_data.get("feasibility", 0.5),
            surprise_score=eval_data.get("surprise", 0.5),
            coherence_score=eval_data.get("coherence", 0.5),
            generativity_score=eval_data.get("generativity", 0.5),
            overall_interest=eval_data.get("overall_interest", 0.5),
            reasoning=eval_data.get("reasoning", ""),
            timestamp=datetime.now().isoformat()
        )
    
    def _compute_novelty(self, idea: Idea, existing_ideas: List[Idea]) -> float:
        """Compute novelty as distance from existing ideas in embedding space"""
        
        if not existing_ideas or idea.embedding is None:
            return 0.8  # Default high novelty if no comparison
        
        # Get embeddings of existing ideas (excluding self)
        existing_embeddings = [
            e.embedding 
            for e in existing_ideas 
            if e.embedding is not None and e.id != idea.id
        ]
        
        if not existing_embeddings:
            return 0.8
        
        # Compute cosine similarity to nearest neighbor
        similarities = cosine_similarity([idea.embedding], existing_embeddings)[0]
        max_similarity = np.max(similarities)
        
        # Novelty is inverse of similarity
        return 1.0 - max_similarity


class IdeaRepository:
    """Stores and queries ideas and evaluations"""
    
    def __init__(self, filepath: str = "idea_repository.json"):
        self.filepath = filepath
        self.ideas: Dict[str, Idea] = {}
        self.evaluations: Dict[str, List[Evaluation]] = {}
        self.load()
    
    def add_idea(self, idea: Idea):
        """Add an idea to repository"""
        self.ideas[idea.id] = idea
        self.evaluations[idea.id] = []
    
    def add_evaluation(self, evaluation: Evaluation):
        """Add an evaluation to repository"""
        if evaluation.idea_id in self.evaluations:
            self.evaluations[evaluation.idea_id].append(evaluation)
    
    def get_top_ideas(self, n: int = 10, metric: str = "overall_interest", problem_filter: Optional[str] = None) -> List[tuple]:
        """Get top N ideas by specified metric, optionally filtered by problem"""
        
        scored_ideas = []
        for idea_id, idea in self.ideas.items():
            # Apply problem filter if specified
            if problem_filter and idea.problem_context != problem_filter:
                continue
                
            evals = self.evaluations.get(idea_id, [])
            if evals:
                # Average score across evaluations
                scores = [getattr(e, f"{metric}_score" if metric != "overall_interest" else "overall_interest") 
                         for e in evals]
                avg_score = np.mean(scores)
                scored_ideas.append((idea, avg_score))
        
        scored_ideas.sort(key=lambda x: x[1], reverse=True)
        return scored_ideas[:n]
    
    def save(self):
        """Save repository to disk"""
        data = {
            "ideas": {k: v.to_dict() for k, v in self.ideas.items()},
            "evaluations": {k: [e.to_dict() for e in v] for k, v in self.evaluations.items()}
        }
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load repository from disk"""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            self.ideas = {k: Idea(**v) for k, v in data.get("ideas", {}).items()}
            self.evaluations = {
                k: [Evaluation(**e) for e in v] 
                for k, v in data.get("evaluations", {}).items()
            }


class IdeaMetabolismSystem:
    """Main orchestrator for the idea generation and evaluation pipeline"""
    
    def __init__(self, llm_provider: str = "anthropic"):
        self.llm = LLMClient(llm_provider)
        self.repository = GraphRepository()
        self.evaluator = IdeaEvaluator(self.llm)
        
        # Initialize persona agents
        self.agents = {
            persona: PersonaAgent(persona, self.llm)
            for persona in PersonaAgent.PERSONAS.keys()
        }
    
    def run_generation_cycle(self, problem: str, ideas_per_persona: int = 3):
        """Run Stage 1: Generate ideas from all personas"""
        
        print(f"\n{'='*60}")
        print("STAGE 1: IDEA GENERATION")
        print(f"{'='*60}\n")
        
        all_ideas = []
        
        for persona_type, agent in self.agents.items():
            print(f"Generating ideas from {agent.config['name']}...")
            ideas = agent.generate_ideas(problem, ideas_per_persona)
            
            # Register problem in graph
            problem_id = self.repository.add_problem(problem)
            
            # Compute embeddings
            for idea in ideas:
                idea.embedding = self.llm.get_embedding(idea.content)
                self.repository.add_idea(idea, problem_id)
            
            all_ideas.extend(ideas)
            print(f"  Generated {len(ideas)} ideas\n")
        
        return all_ideas
    
    def run_triage_cycle(self, ideas: List[Idea]):
        """Run Stage 2: Evaluate and triage ideas"""
        
        print(f"\n{'='*60}")
        print("STAGE 2: TRIAGE & EVALUATION (Graph RAG)")
        print(f"{'='*60}\n")
        
        # RAG: Retrieve context ideas based on current ideas' problem
        # We assume all new ideas share the same problem context for now
        if not ideas:
            return
            
        current_problem = ideas[0].problem_context
        context_dicts = self.repository.get_context_ideas(current_problem)
        
        # Convert dicts to Idea objects for Evaluator compatibility
        # We create a temporary Idea class or simple object
        existing_ideas = []
        for d in context_dicts:
             # Basic reconstruction
             try:
                 obj = Idea(
                     id=d['id'],
                     content=d.get('content',''),
                     persona=d.get('persona',''),
                     temperature=d.get('temperature',0.7),
                     timestamp=d.get('timestamp',''),
                     problem_context=d.get('problem_context',''),
                     embedding=d.get('embedding')
                 )
                 existing_ideas.append(obj)
             except:
                 pass
        
        for idea in ideas:
            print(f"Evaluating idea {idea.id[:20]}...")
            evaluation = self.evaluator.evaluate_idea(idea, existing_ideas)
            self.repository.add_evaluation(evaluation)
            
            print(f"  Novelty: {evaluation.novelty_score:.2f}")
            print(f"  Feasibility: {evaluation.feasibility_score:.2f}")
            print(f"  Surprise: {evaluation.surprise_score:.2f}")
            print(f"  Interest: {evaluation.overall_interest:.2f}\n")
            
        # Extract relationships (NEW)
        self.extract_relationships(ideas, existing_ideas)
        
        self.repository.save()

    def extract_relationships(self, new_ideas: List[Idea], existing_ideas: List[Idea]):
        """Stage 3: Extract semantic relationships between ideas"""
        print(f"\n{'='*60}")
        print("STAGE 3: RELATIONSHIP EXTRACTION")
        print(f"{'='*60}\n")
        
        if not existing_ideas:
            return

        # Simple batch processing for demo purposes
        # Compares each new idea against top 5 most similar existing ideas to save tokens
        
        for idea in new_ideas:
            # Simple retrieval of candidates based on embedding similarity would be better here
            # For now, just compare against first few context ideas
            candidates = existing_ideas[:5] 
            
            prompt = f"""Analyze relationships between this new idea and existing ideas.
            
NEW IDEA: {idea.content}

EXISTING IDEAS:
{json.dumps([{'id': e.id, 'content': e.content[:200]} for e in candidates], indent=2)}

Identify if the NEW IDEA:
1. CONTRADICTS any existing idea (fundamentally incompatible)
2. REQUIRES any existing idea (is a prerequisite or dependency)

Return JSON array of relationships:
[
  {{"target_id": "...", "type": "CONTRADICTS", "reason": "..."}},
  {{"target_id": "...", "type": "REQUIRES", "reason": "..."}}
]
If no relationships, return []."""

            response = self.llm.generate(prompt, temperature=0.1)
            
            try:
                start = response.find('[')
                end = response.rfind(']') + 1
                rels = json.loads(response[start:end])
                
                for rel in rels:
                    target_id = rel.get('target_id')
                    rel_type = rel.get('type')
                    reason = rel.get('reason')
                    
                    if target_id and rel_type in ["CONTRADICTS", "REQUIRES"]:
                        self.repository.add_relationship(idea.id, target_id, rel_type, reason)
                        print(f"  [RELATIONSHIP] {rel_type}: {reason[:100]}...")
            except:
                pass

    def run_evolution_cycle(self, ideas: List[Idea]):
        """Stage 4: Evolutionary Synthesis (Breeding)"""
        print(f"\n{'='*60}")
        print("STAGE 4: EVOLUTIONARY SYNTHESIS")
        print(f"{'='*60}\n")
        
        if not ideas:
            return

        # 1. Selection: Find Novel and Feasible parents from current batch AND history
        
        # Get historical context ideas
        current_problem = ideas[0].problem_context
        context_dicts = self.repository.get_context_ideas(current_problem)
        
        historical_ideas = []
        for d in context_dicts:
             try:
                 obj = Idea(
                     id=d['id'],
                     content=d.get('content',''),
                     persona=d.get('persona',''),
                     temperature=d.get('temperature',0.7),
                     timestamp=d.get('timestamp',''),
                     problem_context=d.get('problem_context',''),
                     embedding=d.get('embedding')
                 )
                 historical_ideas.append(obj)
             except:
                 pass

        # Combine candidates (ensure uniqueness by ID)
        all_candidates = {idea.id: idea for idea in ideas}
        for h_idea in historical_ideas:
            if h_idea.id not in all_candidates:
                all_candidates[h_idea.id] = h_idea
        
        candidate_list = list(all_candidates.values())
        print(f"Breeding Pool Size: {len(candidate_list)} ideas (Current + History)")

        # We need the evaluations for these ideas
        scored_candidates = []
        for idea in candidate_list:
            evals = self.repository.get_evaluations(idea.id)
            if evals:
                # Use first evaluation
                e = evals[0]
                scored_candidates.append({
                    "idea": idea,
                    "novelty": getattr(e, "novelty_score", 0),
                    "feasibility": getattr(e, "feasibility_score", 0)
                })
        
        if len(scored_candidates) < 2:
            print("Not enough evaluated ideas for breeding.")
            return

        # Select Parent A (Most Novel)
        parent_a_data = max(scored_candidates, key=lambda x: x['novelty'])
        parent_a = parent_a_data['idea']
        
        # Select Parent B (Most Feasible) - distinct from A
        remaining = [c for c in scored_candidates if c['idea'].id != parent_a.id]
        if not remaining:
            return
            
        parent_b_data = max(remaining, key=lambda x: x['feasibility'])
        parent_b = parent_b_data['idea']
        
        print(f"Parent A (Novelty {parent_a_data['novelty']:.2f}): {parent_a.content[:50]}...")
        print(f"Parent B (Feasibility {parent_b_data['feasibility']:.2f}): {parent_b.content[:50]}...")
        
        # 2. Crossover & Mutation
        prompt = f"""Perform an evolutionary synthesis of two ideas.
        
PARENT A (High Novelty):
{parent_a.content}

PARENT B (High Feasibility):
{parent_b.content}

PROBLEM: {parent_a.problem_context}

TASK: Generate a "Child Idea" that combines the core novel mechanism of Parent A with the practical grounding of Parent B.
The child should be a distinct concept, not just a concatenation. 
Mutate the idea slightly to ensure it evolves beyond both parents.

Format as JSON:
{{
  "idea": "...",
  "rationale": "...",
  "assumptions": "..."
}}"""

        response = self.llm.generate(prompt, temperature=0.7)
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            child_data = json.loads(response[start:end])
            
            child_text = f"{child_data.get('idea', '')}\n\nRationale: {child_data.get('rationale', '')}\nAssumptions: {child_data.get('assumptions', '')}"
            
            # 3. Integration
            child_id = f"child_{int(time.time())}"
            child_idea = Idea(
                id=child_id,
                content=child_text,
                persona="evolutionary_synthesis",
                temperature=0.7,
                timestamp=datetime.now().isoformat(),
                problem_context=parent_a.problem_context,
                embedding=None
            )
            
            # Embed and Add
            child_idea.embedding = self.llm.get_embedding(child_idea.content)
            
            # Register with graph (using existing problem ID lookup would be better, but we can pass text logic in add_idea or similar)
            # We need the problem ID. 
            # Currently add_idea takes idea and problem_id.
            # We can re-fetch or track problem_id.
            # Let's peek at how we handled it in generation cycle. We retrieved problem_id there.
            # We can lookup problem ID by hashing text or finding the node.
            # GraphRepository doesn't expose lookup easily yet without re-hashing.
            # Let's use `add_problem` which is idempotent and returns ID.
            problem_id = self.repository.add_problem(child_idea.problem_context)
            
            self.repository.add_idea(child_idea, problem_id)
            print(f"\nCreated Child Idea: {child_idea.content[:100]}...")
            
            # Add Lineage Edges
            self.repository.add_relationship(child_id, parent_a.id, "DERIVED_FROM", "Novelty Parent")
            self.repository.add_relationship(child_id, parent_b.id, "DERIVED_FROM", "Feasibility Parent")
            
            # Evaluate Child
            print("Evaluating Child...")
            # We need existing ideas list again.
            # We can pass the full graph context or just the current batch + parents
            # Let's retrieve context again or use 'ideas' list which is the batch
            # Ideally we check against broader context.
            context_dicts = self.repository.get_context_ideas(child_idea.problem_context)
            # Reconstruct... (This logic is duplicated, suggests need for helper)
            existing_for_eval = [] # ... skipped for brevity, let's use the 'ideas' batch + parents
            existing_for_eval.extend(ideas)
            
            evaluation = self.evaluator.evaluate_idea(child_idea, existing_for_eval)
            self.repository.add_evaluation(evaluation)
            
            print(f"  Novelty: {evaluation.novelty_score:.2f}")
            print(f"  Feasibility: {evaluation.feasibility_score:.2f}")
            print(f"  Interest: {evaluation.overall_interest:.2f}\n")
            
        except json.JSONDecodeError:
            print("Failed to generate valid JSON for child idea.")
        except Exception as e:
            print(f"Error during evolution: {e}")
    
    def display_top_ideas(self, n: int = 5, problem: Optional[str] = None):
        """Display top N ideas by overall interest"""
        
        print(f"\n{'='*60}")
        print(f"TOP {n} IDEAS BY OVERALL INTEREST")
        if problem:
            print(f"For problem: {problem}")
        print(f"{'='*60}\n")
        
        top_ideas = self.repository.get_top_ideas(n, problem_filter=problem)
        
        for i, (idea, score) in enumerate(top_ideas, 1):
            print(f"{i}. [{idea.persona.upper()}] Score: {score:.2f}")
            print(f"   {idea.content[:200]}...")
            
            evals = self.repository.get_evaluations(idea.id)
            if evals:
                eval_summary = evals[0]
                print(f"   Reasoning: {eval_summary.reasoning[:150]}...")
            print()
    
    def run(self, problem: str, ideas_per_persona: int = 3):
        """Run complete generation and triage cycle"""
        
        print(f"\nPROBLEM: {problem}\n")
        
        # Stage 1: Generate
        ideas = self.run_generation_cycle(problem, ideas_per_persona)
        
        # Stage 2: Triage
        self.run_triage_cycle(ideas)
        
        # Stage 4: Evolution (NEW)
        self.run_evolution_cycle(ideas)
        
        # Display results
        self.display_top_ideas(problem=problem)
        
        print(f"\nRepository saved to: {self.repository.filepath}")
        print(f"Total ideas in repository: {self.repository.count_ideas()}")


    
    def process_problem(self, problem: str, repo_only: bool = False, limit: int = 5) -> List[Dict]:
        """Process a problem statement and return structured results"""
        
        results = []
        
        if repo_only:
            # Semantic Search Mode
            # Re-using context retrieval logic
            context_dicts = self.repository.get_context_ideas(problem)
            
            # Convert to objects
            found_ideas = []
            for d in context_dicts:
                 try:
                     obj = Idea(
                         id=d['id'],
                         content=d.get('content',''),
                         persona=d.get('persona',''),
                         temperature=d.get('temperature',0.7),
                         timestamp=d.get('timestamp',''),
                         problem_context=d.get('problem_context',''),
                         embedding=d.get('embedding')
                     )
                     found_ideas.append(obj)
                 except:
                     pass
            
            # Formatting results
            for i, idea in enumerate(found_ideas):
                if i >= limit:
                    break
                
                score = 0
                reasoning = ""
                evals = self.repository.get_evaluations(idea.id)
                if evals:
                    score = getattr(evals[0], 'overall_interest', 0)
                    reasoning = evals[0].reasoning
                
                results.append({
                    "id": idea.id,
                    "persona": idea.persona,
                    "content": idea.content,
                    "score": score,
                    "reasoning": reasoning,
                    "source": "repository"
                })
                
        else:
            # Generation Mode
            # Run the full pipeline
            # Suppress stdout to keep API clean? Or just let it print to server logs.
            self.run(problem, ideas_per_persona=3)
            
            # Retrieve top results
            top_ideas = self.repository.get_top_ideas(limit, problem_filter=problem)
            
            for idea, score in top_ideas:
                reasoning = ""
                evals = self.repository.get_evaluations(idea.id)
                if evals:
                    reasoning = evals[0].reasoning
                
                results.append({
                    "id": idea.id,
                    "persona": idea.persona,
                    "content": idea.content,
                    "score": score,
                    "reasoning": reasoning,
                    "source": "generated"
                })
                
        return results

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Idea Metabolism System")
    parser.add_argument("-p", "--problem", type=str, help="The problem statement to address")
    parser.add_argument("-r", "--repo-only", nargs="?", const=5, type=int, 
                        help="Only query repository for existing ideas related to problem (default 5 results)")
    
    args = parser.parse_args()
    
    if not args.problem:
        parser.print_help()
        sys.exit(1)
        
    problem = args.problem
    
    # Initialize system
    system = IdeaMetabolismSystem(llm_provider="anthropic")
    
    # Run using the new API method logic for consistent behavior
    # Note: run() inside process_problem prints logs, which is fine for CLI.
    # The return value also lets us print the final list if we wanted to replace display_top_ideas,
    # but display_top_ideas is nice for CLI formatting.
    
    if args.repo_only:
        print(f"\nsearching repository for: {problem}")
        print(f"Limit: {args.repo_only} ideas\n")
        
        results = system.process_problem(problem, repo_only=True, limit=args.repo_only)
        
        print(f"Found {len(results)} relevant ideas in context.\n")
        
        for i, res in enumerate(results, 1):
             print(f"{i}. [{res['persona'].upper()}] Score: {res['score']:.2f}")
             print(f"   {res['content'][:200]}...")
             if res['reasoning']:
                 print(f"   Reasoning: {res['reasoning'][:150]}...")
             print()
    else:
        # Standard run prints its own output via display_top_ideas inside run()
        # We can just call process_problem to exercise that path if we want, 
        # or stick to system.run() which is what process_problem calls.
        # To ensure process_problem is 'the' API, let's use it, but ignored return for CLI output 
        # since run() prints everything.
        system.process_problem(problem, repo_only=False)
