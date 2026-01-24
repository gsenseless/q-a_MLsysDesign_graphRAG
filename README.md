# ML System Design Repository Agent (GraphRAG)

This project is a GraphRAG-based agent designed to answer questions about ML system design based on the [ML System Design](https://github.com/ML-SystemDesign/MLSystemDesign) repository's markdown files.



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


## Evaluation

The repository features a robust evaluation framework to benchmark agent performance.

### Running Evaluation:

```bash
uv run python src/eval.py
```

### How it Works:

The evaluation script (`src/eval.py`) performs the following steps:

1. **Generate Questions**: Automatically creates test questions based on repository content using an LLM (Mistral).
2. **Run Benchmark**: Executes the agent on the generated questions and logs the interactions to the `logs/` directory.
3. **Analyze Results**: Uses an **LLM-as-a-Judge** approach (with an independent `eval_agent`) to evaluate the agent's responses against a predefined checklist and prints the performance metrics.

#### Metrics Evaluated:
- **factually_grounded**: Checks if the answer is supported by retrieved context.
- **key_information_retrieved**: Checks if the agent missed direct answers present in the context.
- **search_relevance**: Evaluates if the agent's search queries were relevant to the question.
- **formatting_compliance**: Checks for proper Markdown structure.

#### Example output:

```bash
============================================================
FINAL EVALUATION REPORT
Total Questions Evaluated: 48
------------------------------------------------------------
                   Metric Score
       factually_grounded 97.9%
key_information_retrieved 97.9%
         search_relevance 97.9%
        citation_accuracy 14.6%
    formatting_compliance 93.8%
============================================================
```

## Why GraphRAG?

Initial attempts using standard Vector RAG proved insufficient for navigating the repository effectively. For example, when asking *"What are the main sections in ML system design doc?"*, a standard RAG system typically retrieves information from `bookOutline.md`. While this file seems relevant, the correct answer is actually located in the example template file.

Standard RAG fails in these cases because it lacks awareness of the folder/file structure and does not even know the names of the files it is searching. This lack of structural context is why the project transitioned to GraphRAG, enabling the agent to understand the repository's organization and retrieve the most accurate information.


