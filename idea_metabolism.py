"""
Idea Metabolism System: Stage 1 (Generation) & Stage 2 (Triage)

A multi-agent system for generating and evaluating novel ideas using
diverse personas and building a queryable knowledge repository.

Usage:
    python idea_metabolism.py --problem "your problem statement"
    
Requirements:
    pip install anthropic openai google-generativeai numpy scikit-learn
"""

import os
import json
import time
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Literal
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


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
        self.repository = IdeaRepository()
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
            
            # Compute embeddings
            for idea in ideas:
                idea.embedding = self.llm.get_embedding(idea.content)
                self.repository.add_idea(idea)
            
            all_ideas.extend(ideas)
            print(f"  Generated {len(ideas)} ideas\n")
        
        return all_ideas
    
    def run_triage_cycle(self, ideas: List[Idea]):
        """Run Stage 2: Evaluate and triage ideas"""
        
        print(f"\n{'='*60}")
        print("STAGE 2: TRIAGE & EVALUATION")
        print(f"{'='*60}\n")
        
        existing_ideas = list(self.repository.ideas.values())
        
        for idea in ideas:
            print(f"Evaluating idea {idea.id[:20]}...")
            evaluation = self.evaluator.evaluate_idea(idea, existing_ideas)
            self.repository.add_evaluation(evaluation)
            
            print(f"  Novelty: {evaluation.novelty_score:.2f}")
            print(f"  Feasibility: {evaluation.feasibility_score:.2f}")
            print(f"  Surprise: {evaluation.surprise_score:.2f}")
            print(f"  Interest: {evaluation.overall_interest:.2f}\n")
        
        self.repository.save()
    
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
            
            evals = self.repository.evaluations[idea.id]
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
        
        # Display results
        self.display_top_ideas(problem=problem)
        
        print(f"\nRepository saved to: {self.repository.filepath}")
        print(f"Total ideas in repository: {len(self.repository.ideas)}")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Get problem from command line or use default
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
    else:
        problem = """How can we create machine learning systems that generate truly novel ideas 
        rather than just recombining existing patterns from training data?"""
    
    # Initialize system
    system = IdeaMetabolismSystem(llm_provider="anthropic")
    
    # Run generation and triage
    system.run(problem, ideas_per_persona=3)
