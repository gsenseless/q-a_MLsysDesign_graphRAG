import os
from datetime import datetime
from typing import Any

import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer

from chunking import process_repo_chunks
from get_repo_data import read_repo_data
from search import create_vector_index


from pydantic_ai.models.openai import OpenAIModel


def create_repo_agent(docs_vindex, embedding_model):
    def get_context(query: str) -> list[Any]:
        """
        Retrieve relevant technical fragments from the repository.

        Args:
            query (str): The query string.

        Returns:
            List[Any]: A list of up to 5 relevant fragments.
        """
        from search import vector_search

        return vector_search(query, docs_vindex, embedding_model)

    system_prompt = """
    You are a helpful assistant for a ML system design repository. 

    Use the available tools to find relevant information from the repository before answering questions.

    If you can find specific information, use it to provide accurate answers.
    If no relevant results are found, let the user know and provide general guidance.
    """

    agent = Agent(
        model="mistral:mistral-small-latest",
        system_prompt=system_prompt,
        tools=[get_context],
    )

    return agent


def generate_report(result: Any, query: str) -> str:
    """
    Generate a detailed markdown report for the agent run.
    """
    messages = []
    final_output = None

    if hasattr(result, "new_messages"):
        messages = result.new_messages()
        final_output = getattr(result, "output", getattr(result, "data", None))
    elif isinstance(result, list):
        messages = result
    else:
        return f"Could not generate report for type {type(result)}"

    report = []
    report.append("# Agent Run Report")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("## 1. Question")
    report.append(f"> {query}")
    report.append("")

    report.append("## 2. Relevant Chunks & Search Chain")

    found_chunks = False
    for message in messages:
        parts = getattr(message, "parts", [])
        for part in parts:
            kind = getattr(part, "part_kind", "unknown")
            if kind == "tool-return":
                content = getattr(part, "content", None)
                if isinstance(content, list):
                    found_chunks = True
                    for i, item in enumerate(content):
                        if isinstance(item, dict):
                            report.append(f"### Chunk {i + 1}")

                            # Extract details
                            chunk_text = item.get("chunk", "")
                            filename = item.get("filename", "unknown")
                            folder = item.get("folder", "unknown")
                            score = item.get("score", 0.0)
                            chunk_score = item.get("chunk_score", 0.0)
                            folder_bonus = item.get("folder_bonus", 0.0)
                            file_bonus = item.get("file_bonus", 0.0)

                            # Format chain
                            report.append(
                                f"**Chain:** `(Folder: {folder})` -> `(File: {filename})` -> `(Chunk)`"
                            )
                            report.append(
                                f"**Score Details:** Total: {score:.4f} (Chunk: {chunk_score:.4f} + Folder Bonus: {folder_bonus * 0.5:.4f} + File Bonus: {file_bonus * 0.5:.4f})"
                            )

                            report.append("```text")
                            report.append(chunk_text.strip())
                            report.append("```")
                            report.append("")

    if not found_chunks:
        report.append("_No chunks found or tool not called._")
    report.append("")

    report.append("## 3. Model Answer")
    if final_output:
        report.append(str(final_output))
    else:
        # Try to find the last text part
        last_text = ""
        for message in reversed(messages):
            parts = getattr(message, "parts", [])
            for part in parts:
                if getattr(part, "part_kind", "") == "text":
                    last_text = getattr(part, "content", "")
                    break
            if last_text:
                break
        report.append(last_text if last_text else "_No answer generated._")

    return "\n".join(report)


if __name__ == "__main__":
    load_dotenv(".env")

    ml_system_design_repo = read_repo_data("ML-SystemDesign", "MLSystemDesign")
    print(f"Repo length: {len(ml_system_design_repo)}")

    ml_system_design_chunks = process_repo_chunks(
        ml_system_design_repo, "sliding_window"
    )
    print(f"Chunks length: {len(ml_system_design_chunks)}")

    embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    docs_vindex = create_vector_index(ml_system_design_chunks)

    ### -----
    agent = create_repo_agent(docs_vindex, embedding_model)

    #    question = "list essential sections of ml system design doc?"
    question = "typical chapters in ML sys design doc"

    result = agent.run_sync(user_prompt=question)

    print(result)

    # Generate and save report
    report_content = generate_report(result, question)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    report_filename = os.path.join(reports_dir, f"report_{timestamp}.md")
    with open(report_filename, "w") as f:
        f.write(report_content)
    print(f"\nReport saved to {report_filename}")
