# Idea Metabolism POC

A multi-agent AI system for generating, evaluating, and evolving ideas over time. This Proof of Concept (POC) demonstrates a "metabolic" approach to ideation where ideas are not just generated once but are stored, retrieved, and critiqued in a persistent knowledge graph.

## Purpose

Standard LLM interactions are ephemeral: you ask for ideas, get a list, and the context vanishes. **Idea Metabolism** treats ideas as persistent entities that reside in a "collective memory."

Key capabilities:
*   **Diverse Personas**: "Convergent", "Divergent", and "Alternative" agents generate distinct types of ideas.
*   **Persistent Memory**: Ideas are stored in a Graph RAG repository, allowing the system to recall past solutions.
*   **Semantic Novelty**: The system calculates how "new" an idea is by comparing its vector embedding against the existing knowledge graph.
*   **Relationship Mapping**: Automatically detects if new ideas **CONTRADICT** or **REQUIRE** existing ideas.

## Architecture

### Core Components
*   **`idea_metabolism.py`**: The main orchestrator. Manages the LLM client, agents, and the 3-stage pipeline (Generation -> Triage -> Relationship Extraction).
*   **`graph_repository.py`**: A **NetworkX**-based graph database.
    *   **Nodes**: Problems, Ideas, Domains.
    *   **Edges**: `ADDRESSES`, `RELATES_TO`, `CONTRADICTS`, `REQUIRES`.
    *   **Vector Search**: Uses `sentence-transformers` for semantic retrieval of problem contexts.

### Data Flow
1.  **Generation**: Agents generate ideas for a given Problem.
2.  **Graph RAG**: The system embeds the Problem and retrieves semantically similar "Context Ideas" from the graph.
3.  **Evaluation**: An Evaluator Agent scores new ideas on Novelty, Feasibility, and Surprise relative to the retrieved context.
4.  **Linking**: The system identifies and creates semantic edges between new and existing ideas.

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
```

The system will:
1.  Generate ~9 ideas from different personas.
2.  Compare them against the graph history.
3.  Output the Top 5 ideas with scores and reasoning.
4.  Save the updated graph to `idea_graph.gml`.
