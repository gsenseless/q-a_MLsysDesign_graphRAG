# ML System Design Repository Agent (GraphRAG)

This project is a GraphRAG-based agent designed to answer questions about ML system design based on a GitHub repository's markdown files.

## Prerequisites

- **Python 3.12+**
- **Docker** (for running Neo4j)
- **uv** (recommended for package management)

## Setup

### 1. Environment Variables

Create a `.env` file in the root directory and add your credentials. You can use `.env.example` as a template:

```bash
cp .env.example .env
```

Edit `.env` to include your `MISTRAL_API_KEY`.

### 2. Run Neo4j Database

You can start the Neo4j database using Docker Compose:

```bash
docker compose up -d
```

Or using a direct docker run command:

```bash
docker run \
    -d \
    --name neo4j_rag \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

Wait a few seconds for the database to initialize before running the application.

### 3. Install Dependencies

Using `uv`:

```bash
uv sync
```

Or using standard pip (if not using uv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Running the Agent

To run the main agent script:

```bash
uv run src/agent.py
```

## Project Structure

- `src/agent.py`: Main agent logic using PydanticAI.
- `src/search.py`: Neo4j GraphIndex implementation for GraphRAG.
- `src/chunking.py`: Document processing and chunking logic.
- `src/get_repo_data.py`: GitHub repository data retrieval.
- `src/eval.py`: Evaluation scripts (if applicable).
