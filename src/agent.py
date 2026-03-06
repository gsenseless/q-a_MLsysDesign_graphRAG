import os
from datetime import datetime
from typing import Any

import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer

from chunking import process_repo_chunks
from get_repo_data import read_repo_data
from reports import generate_report
from search import create_vector_index, vector_search


def create_repo_agent(docs_vindex, embedding_model):
    """Create an agent with repository search capabilities."""

    def get_context(query: str) -> list[Any]:
        """
        Retrieve relevant technical fragments from the repository.

        Args:
            query (str): The query string.

        Returns:
            List[Any]: A list of up to 5 relevant fragments.
        """

        return vector_search(query, docs_vindex, embedding_model)

    system_prompt = """
        You are a helpful assistant for a ML system design repository.

        Use the available tools to find relevant information from the repository before answering questions.

        If you can find specific information, use it to provide accurate answers.
        If no relevant results are found, let the user know and provide general guidance.
        """

    return Agent(
        model="mistral:mistral-small-latest",
        system_prompt=system_prompt,
        tools=[get_context],
    )

    return agent

def setup_repository(repo_name: str, repo_folder: str):
    """Load repository data and create vector index."""
    repo_data = read_repo_data(repo_name, repo_folder)
    print(f"Repo length: {len(repo_data)}")

    chunks = process_repo_chunks(repo_data, "sliding_window")
    print(f"Chunks length: {len(chunks)}")

    embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    docs_vindex = create_vector_index(chunks)

    return docs_vindex, embedding_model


def run_agent_query(agent, question: str) -> Any:
    """Run agent with question and return result."""
    result = agent.run_sync(user_prompt=question)
    print(result)
    return result


def save_report(report_content: str, reports_dir: str = "reports") -> str:
    """Save report to file and return filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(reports_dir, exist_ok=True)
    report_filename = os.path.join(reports_dir, f"report_{timestamp}.md")
    with open(report_filename, "w") as f:
        f.write(report_content)
    return report_filename


if __name__ == "__main__":
    load_dotenv(".env")

    docs_vindex, embedding_model = setup_repository("ML-SystemDesign", "MLSystemDesign")
    agent = create_repo_agent(docs_vindex, embedding_model)

    question = "typical chapters in ML sys design doc"
    result = run_agent_query(agent, question)

    report_content = generate_report(result, question)
    report_filename = save_report(report_content)
    print(f"\nReport saved to {report_filename}")
