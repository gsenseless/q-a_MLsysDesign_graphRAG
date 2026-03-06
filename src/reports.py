from datetime import datetime
from typing import Any


def format_chunk(chunk_data: dict, index: int) -> str:
    """Format a single chunk as markdown."""
    filename = chunk_data.get("filename", "unknown")
    folder = chunk_data.get("folder", "unknown")
    score = chunk_data.get("score", 0.0)
    chunk_score = chunk_data.get("chunk_score", 0.0)
    folder_bonus = chunk_data.get("folder_bonus", 0.0)
    file_bonus = chunk_data.get("file_bonus", 0.0)
    chunk_text = chunk_data.get("chunk", "")

    return f"""### Chunk {index + 1}
**Chain:** `(Folder: {folder})` -> `(File: {filename})` -> `(Chunk)`
**Score Details:** Total: {score:.4f} (Chunk: {chunk_score:.4f} + Folder Bonus: {folder_bonus * 0.5:.4f} + File Bonus: {file_bonus * 0.5:.4f})
```text
{chunk_text.strip()}
```
"""


def extract_chunks_from_messages(messages: list) -> str:
    """Extract and format chunks from message history."""
    chunks_found = False
    sections = []

    for message in messages:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") != "tool-return":
                continue
            content = getattr(part, "content", None)
            if not isinstance(content, list):
                continue
            chunks_found = True
            for i, item in enumerate(content):
                if isinstance(item, dict):
                    sections.append(format_chunk(item, i))

    return "\n".join(sections) if chunks_found else "_No chunks found or tool not called._"


def extract_final_answer(messages: list, final_output: Any = None) -> str:
    """Extract the final answer from result."""
    if final_output:
        return str(final_output)

    for message in reversed(messages):
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") == "text":
                content = getattr(part, "content", "")
                if content:
                    return content
    return "_No answer generated._"


def generate_report(result: Any, query: str) -> str:
    """Generate a detailed markdown report for the agent run."""
    if hasattr(result, "new_messages"):
        messages = result.new_messages()
        final_output = getattr(result, "output", getattr(result, "data", None))
    elif isinstance(result, list):
        messages = result
        final_output = None
    else:
        return f"Could not generate report for type {type(result)}"

    chunks_section = extract_chunks_from_messages(messages)
    answer_section = extract_final_answer(messages, final_output)

    return f"""# Agent Run Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Question
> {query}

## 2. Relevant Chunks & Search Chain

{chunks_section}

## 3. Model Answer
{answer_section}
"""
