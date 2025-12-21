# Idea Metabolism POC

A Persistent Creative Intelligence System that Learns and Evolves Ideas Over Time. Unlike traditional AI that forgets each conversation, this system builds a living memory of ideas that grow smarter through specialized AI personas, semantic analysis, and evolutionary breeding of concepts—transforming one-shot brainstorming into continuous creative intelligence.

## Purpose

Standard LLM interactions are ephemeral: you ask for ideas, get a list, and the context vanishes. **Idea Metabolism** treats ideas as persistent entities that reside in a "collective memory". Knowledge accumulates in the collective memory, allowing the system to generate, evaluate, and evolve ideas over time as new information becomes available. Inspired by human creative processes, the system can generate, evaluate, and evolve ideas over time as new information becomes available. This POC is a step in overcoming the backward-looking nature of LLMs which are subject to representational collapse optimizing for known distributions and penalizing divergence thus stifling innovation.

Key capabilities:
*   **Diverse Personas**: "Convergent", "Divergent", and "Alternative" agents generate distinct types of ideas.
*   **Persistent Memory**: Ideas are stored in a Graph RAG repository, allowing the system to recall past solutions.
*   **Semantic Novelty**: The system calculates how "new" an idea is by comparing its vector embedding against the existing knowledge graph.
*   **Relationship Mapping**: Automatically detects if new ideas **CONTRADICT** or **REQUIRE** existing ideas.
*   **Evolutionary Synthesis**: Actively "breeds" new ideas by combining the most *Novel* idea with the most *Feasible* idea from the current generation cycle **and** relevant history, creating offspring that inherit traits from both.

## Architecture

### Core Components
*   **`idea_metabolism.py`**: The main orchestrator. Manages the LLM client, agents, and the 4-stage pipeline (Generation -> Triage -> Relationship Extraction -> Evolution).
*   **`graph_repository.py`**: A **NetworkX**-based graph database.
    *   **Nodes**: Problems, Ideas, Domains.
    *   **Edges**: `ADDRESSES`, `RELATES_TO`, `CONTRADICTS`, `REQUIRES`, `DERIVED_FROM`.
    *   **Vector Search**: Uses `sentence-transformers` for semantic retrieval of problem contexts.

### Data Flow
1.  **Generation**: Agents generate ideas for a given Problem.
2.  **Graph RAG**: The system embeds the Problem and retrieves semantically similar "Context Ideas" from the graph.
3.  **Evaluation**: An Evaluator Agent scores new ideas on Novelty, Feasibility, and Surprise relative to the retrieved context.
4.  **Linking**: The system identifies and creates semantic edges between new and existing ideas.
5.  **Evolution**: The system identifies the "Most Novel" and "Most Feasible" ideas in the batch (plus relevant historical ideas) and synthesizes a "Child Idea" that attempts to maximize both traits. This child is then evaluated and stored with `DERIVED_FROM` lineage edges.

## Setup & Usage

### Prerequisites
*   Python 3.10+
*   An API Key for **Anthropic** (default), **OpenAI**, or **Google Gemini**.

### Installation

1.  **Clone the repository** (or navigate to directory).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Dependencies include: `anthropic`, `openai`, `google-generativeai`, `numpy`, `scikit-learn`, `sentence-transformers`, `networkx`, `python-dotenv`.*

3.  **Environment Setup**:
    Create a `.env` file in the root directory:
    ```bash
    ANTHROPIC_API_KEY=sk-ant-...
    # Optional:
    # OPENAI_API_KEY=sk-...
    # GOOGLE_API_KEY=...
    ```

### Running the System

To generate ideas for a specific problem:

```bash
python idea_metabolism.py --problem "How to reduce plastic waste in oceans?"
# OR
python idea_metabolism.py -p "How to reduce plastic waste in oceans?" -n 2
```

**Options:**
*   `-p`, `--problem`: The problem statement (required).
*   `-n`, `--ideas-per-persona`: Number of ideas to generate per persona (default: 1).
*   `-r`, `--repo-only`: Search existing ideas instead of generating. Accepts an optional integer for the max number of results (default: 5).

To query the repository for similar existing ideas (skip generation):

```bash
# Get top 5 relevant ideas
python idea_metabolism.py -p "How to reduce plastic waste?" --repo-only

# Get top 3 relevant ideas
python idea_metabolism.py -p "How to reduce plastic waste?" -r 3
```

### Web Interface

The system includes a Flask-based mobile-friendly web interface.

1.  **Start the Server**:
    ```bash
    python application.py
    ```
    The server will start on `http://0.0.0.0:5000`.

2.  **Access**:
    Open your browser and navigate to `http://localhost:5000` (or your machine's IP address).

    *   **Repo Only Checked**: Search existing ideas.
    *   **Repo Only Unchecked**: Generate new ideas (configure "Ideas per Persona" to control volume).

The system will:
1.  Generate ~3 ideas from different personas (plus 1 synthesized from evolutionary cross-breeding).
2.  Compare them against the graph history.
3.  Output the Top 5 ideas with scores and reasoning.
4.  Save the updated graph to `idea_graph.gml`.
