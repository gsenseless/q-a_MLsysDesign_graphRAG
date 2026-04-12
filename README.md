# ML System Design Repository Agent (GraphRAG)

This project is a GraphRAG-based agent designed to answer questions about ML system design based on the [ML System Design](https://github.com/ML-SystemDesign/MLSystemDesign) repository's markdown files.



## Prerequisites

- **Python 3.12+**
- **Docker** (for running Neo4j)
- **uv** (package management)

## Setup

### 1. Environment Variables

Create a `.env` file in the root directory and add your credentials. You can use `.env.example` as a template:

```bash
cp .env.example .env
```

Edit `.env` to include your `MISTRAL_API_KEY`.

### 2. Run Neo4j Database

Using Docker Compose:

```bash
docker compose up -d
```

Wait a few seconds for the database to initialize before running the application.

### 3. Install Dependencies

Using `uv`:

```bash
uv sync
```

## Streamlit UI (Usage)

The project includes a Streamlit-based web interface for an interactive Q&A experience.

To run the Streamlit app:

```bash
uv run streamlit run src/app.py
```


<p align="center">
  <img src="assets/images/image.png" width="500" alt="Streamlit UI">
</p>

Navigate to `http://localhost:8501` in your web browser to access the Streamlit interface.

Ask a question and view the sources used to generate the answer:

<p align="center">
  <img src="assets/images/image-1.png" width="400" alt="Source Attribution">
</p>


## How it works [Under the Hood]

The system operates as an autonomous agent rather than a traditional RAG pipeline.

- **Agentic Decision Making**: Built with `pydantic_ai`, the agent doesn't just "receive" context. It is equipped with a `get_context` tool which it calls autonomously to retrieve repository fragments based on the user's query.
- **Hybrid Search**: When `get_context` is called, it executes an optimized **Cypher query** in Neo4j that performs:
    - **Tri-partite Retrieval**: Searches vector indexes for relevant folders, files, and chunks in a single pass.
    - **Graph-Based Re-ranking**: Automatically "boosts" the score of text chunks if they belong to a folder or file that is also semantically relevant, leveraging the structural links in the knowledge graph.

## Evaluation

The repository features a robust evaluation framework to benchmark agent performance.

### Running Evaluation:

```bash
uv run python src/eval.py
```

### How it Works:

The evaluation runs in two phases:

**Phase 1 — Generate logs**
1. Loads the ML-SystemDesign repository, processes it into chunks, and builds a vector index.
2. An LLM generates realistic test questions from sampled repository content.
3. The repo agent answers each question; the full interaction (search queries, retrieved chunks, answer) is saved to `logs/`.

**Phase 2 — Judge logs**
4. A separate `eval_agent` (LLM-as-a-Judge) reads each saved log — including the actual retrieved chunks — and scores the agent's response across 5 metrics.
5. Scores are averaged and printed as a final report.

```mermaid
flowchart TD
    A[read_repo_data] --> B[process_repo_chunks]
    B --> C[create_vector_index]
    C --> D[create_repo_agent]
    D --> E["question_generator<br>generates questions from<br>sampled repo docs"]
    E --> F["repo_agent runs on<br>each question"]
    F --> G["logs saved to logs/*.json<br>question · tool calls · answer"]

    G --> H[load_evaluation_set]
    H --> I["simplify_log_messages<br>strips metadata, keeps chunks"]
    I --> J["eval_agent<br>LLM-as-a-Judge<br>scores 5 metrics per log"]
    J --> K[FINAL EVALUATION REPORT]

    style G fill:#f0f0f0,stroke:#999
    style K fill:#d4edda,stroke:#28a745
```

#### Metrics Evaluated:
- **factually_grounded**: Checks if the answer is consistent with the retrieved chunks (no hallucinated facts).
- **key_information_retrieved**: Checks if the agent missed a direct answer that was present in the retrieved chunks.
- **search_relevance**: Evaluates if the agent's search queries matched the concepts in the question.
- **citation_accuracy**: Checks if the answer cites specific source filenames (e.g. `CONTRIBUTING.md`) rather than vague phrases like "the repository".
- **formatting_compliance**: Checks for appropriate Markdown structure (bullets, bolding).

#### Example output:

```bash
============================================================
FINAL EVALUATION REPORT
Total Questions Evaluated: 44
------------------------------------------------------------
                   Metric Score
       factually_grounded  97.7%
key_information_retrieved  95.5%
         search_relevance 100.0%
        citation_accuracy  38.6%
    formatting_compliance 100.0%
============================================================
```

## Why GraphRAG?

Initial attempts using standard Vector RAG proved insufficient for navigating the repository effectively. For example, when asking *"What are the main sections in ML system design doc?"*, a standard RAG system typically retrieves information from `bookOutline.md`. While this file seems relevant, the correct answer is actually located in the example template file.

Standard RAG fails in these cases because it lacks awareness of the folder/file structure and does not even know the names of the files it is searching. This lack of structural context is why the project transitioned to GraphRAG, enabling the agent to understand the repository's organization and retrieve the most accurate information.


