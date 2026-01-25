# Idea Metabolism POC

A Persistent Creative Intelligence System that Learns and Evolves Ideas Over Time. Unlike traditional AI that forgets each conversation, this system builds a living memory of ideas that grow smarter through specialized AI personas, semantic analysis, human feedback, and evolutionary breeding of concepts—transforming one-shot brainstorming into continuous creative intelligence.

## Purpose

Standard LLM interactions are ephemeral: you ask for ideas, get a list, and the context vanishes. **Idea Metabolism** treats ideas as persistent entities that reside in a "collective memory". Knowledge accumulates in the collective memory, allowing the system to generate, evaluate, and evolve ideas over time as new information becomes available. Inspired by human creative processes, the system can generate, evaluate, and evolve ideas over time as new information becomes available. This POC is a step in overcoming the backward-looking nature of LLMs which are subject to representational collapse optimizing for known distributions and penalizing divergence thus stifling innovation.

Key capabilities:
*   **Diverse Personas**: "Convergent", "Divergent", and "Alternative" agents generate distinct types of ideas.
*   **Persistent Memory**: Ideas are stored in a Graph RAG repository, allowing the system to recall past solutions.
*   **Semantic Novelty**: The system calculates how "new" an idea is by comparing its vector embedding against existing knowledge. It uses a **k-Nearest Neighbor** approach to prevent duplicates while respecting problem-specific contexts.
*   **Relationship Mapping**: Automatically detects if new ideas **CONTRADICT** or **REQUIRE** existing ideas.
*   **Evolutionary Synthesis**: Actively "breeds" new ideas by combining the most *Novel* (based on k-NN score) idea with the highest *Interest* idea from the current generation cycle **and** relevant history, creating offspring that inherit traits from both.
*   **Human Feedback Loop**: Users can rate ideas (1-5 stars) and leave comments. Ratings boost/penalize scores, influencing which ideas are selected for breeding.
*   **Lineage Visualization**: View parent-child relationships and semantic connections for any idea.

## Architecture

### Core Components
*   **`idea_metabolism.py`**: The main orchestrator. Manages the LLM client, agents, and the 4-stage pipeline (Generation → Triage → Relationship Extraction → Evolution).
*   **`graph_repository.py`**: A **NetworkX**-based graph database.
    *   **Nodes**: Problems, Ideas, Domains.
    *   **Edges**: `ADDRESSES`, `RELATES_TO`, `CONTRADICTS`, `REQUIRES`, `DERIVED_FROM`.
    *   **Vector Search**: Uses `sentence-transformers` for semantic retrieval of problem contexts.
    *   **Score Breakdown**: Provides AI score, human feedback stats, and combined interest score.
*   **`application.py`**: Flask web server with API endpoints for idea generation, feedback, and individual idea pages.

### Data Flow
1.  **Generation**: Agents generate ideas for a given Problem.
2.  **Graph RAG**: The system embeds the Problem and retrieves semantically similar "Context Ideas" from the graph.
3.  **Evaluation**: An Evaluator Agent scores new ideas on Novelty, Feasibility, and Surprise relative to the retrieved context.
4.  **Linking**: The system identifies and creates semantic edges between new and existing ideas.
5.  **Evolution**: The system identifies the "Most Novel" and "Highest Interest" ideas in the batch (plus relevant historical ideas) and synthesizes a "Child Idea" that attempts to maximize novelty while preserving proven appeal. This child is then evaluated and stored with `DERIVED_FROM` lineage edges.
6.  **User Feedback**: Humans rate and comment on ideas. High ratings boost an idea's selection probability for future breeding.

### Scoring System

Ideas are ranked by **Combined Interest Score**:
- **AI Interest**: Base score from LLM evaluation (novelty × feasibility × surprise)
- **Human Feedback Multiplier**:
  - 4.5+ stars → +50%
  - 4.0+ stars → +25%
  - 3.0+ stars → +10%
  - 2.0+ stars → -20%
  - Below 2.0 → -50%
47: 
### Novelty Calculation
The system uses a dynamic, problem-scoped novelty score:
1.  **Dynamic**: Calculated at query time against the *current* state of the database, so scores update as new ideas are added.
2.  **Problem-Scoped**: An idea is only compared against other ideas that address the *same* problem. Ideas in different domains (e.g., "Cooking" vs "Transport") do not affect each other's novelty.
3.  **k-NN Similarity**: Instead of checking for a single duplicate (Max Similarity), the system calculates the **Average Similarity of the top 5 nearest neighbors**. This ensures that:
    -   Single outliers don't artificially tank a score.
    -   Clusters of similar ideas progressively lower the novelty for all members of that cluster.


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
    *Dependencies include: `anthropic`, `openai`, `google-generativeai`, `numpy`, `scikit-learn`, `sentence-transformers`, `networkx`, `python-dotenv`, `filelock`.*

3.  **Environment Setup**:
    Create a `.env` file in the root directory:
    ```bash
    ANTHROPIC_API_KEY=sk-ant-...
    # Optional:
    # OPENAI_API_KEY=sk-...
    # GOOGLE_API_KEY=...
    # BASE_URL=https://your-domain.com  # For shareable idea links
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

    *   **Strategy Dropdown**: Choose "Mix" (new + history), "New Only", or "Search Memory Only".
    *   Click on any idea card to open the detail modal with rating, comment, and share options.

3.  **Individual Idea Pages** (`/idea/<id>`):
    *   View full idea content and AI evaluation
    *   See score breakdown (AI Interest, User Feedback, Combined Score)
    *   Explore **Lineage** (parent/child relationships) and **Other Connections** (semantic relationships)
    *   Share via X, LinkedIn, or copy link

The system will:
1.  Generate ~3 ideas from different personas (plus 1 synthesized from evolutionary cross-breeding).
2.  Compare them against the graph history.
3.  Output the Top 5 ideas with scores and reasoning.
4.  Save the updated graph to `idea_graph.gml`.
