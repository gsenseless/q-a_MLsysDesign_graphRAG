import json
import secrets
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter


def serializer(obj: object) -> str:
    """Serialize datetime objects to ISO format."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def log_entry(agent, messages, source: str = "user") -> dict:
    """Create a log entry from agent interaction."""
    tools = []

    for ts in agent.toolsets:
        tools.extend(ts.tools.keys())

    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    return {
        "agent_name": agent.name,
        "system_prompt": agent._instructions,
        "provider": agent.model.system,
        "model": agent.model.model_name,
        "tools": tools,
        "messages": dict_messages,
        "source": source,
    }


def log_interaction_to_file(agent, messages, log_dir: Path, source: str = "user") -> Path:
    """Log an agent interaction to a JSON file."""
    entry = log_entry(agent, messages, source)

    ts = entry["messages"][-1]["timestamp"]
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
    filepath = log_dir / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    return filepath


def load_log_file(log_file: Path | str) -> dict:
    """Load a log file and add the filepath to the data."""
    with Path(log_file).open() as f_in:
        log_data = json.load(f_in)
        log_data["log_file"] = Path(log_file)
        return log_data


def simplify_log_messages(messages: list) -> list:
    """Simplify log messages by removing redundant fields."""
    log_simplified = []

    for m in messages:
        parts = []

        for original_part in m["parts"]:
            part = original_part.copy()
            kind = part["part_kind"]

            if kind == "user-prompt":
                del part["timestamp"]
            if kind == "tool-call":
                del part["tool_call_id"]
            if kind == "tool-return":
                del part["tool_call_id"]
                del part["metadata"]
                del part["timestamp"]
                # Replace actual search results with placeholder to save tokens
                part["content"] = "RETURN_RESULTS_REDACTED"
            if kind == "text":
                del part["id"]

            parts.append(part)

        message = {"kind": m["kind"], "parts": parts}

        log_simplified.append(message)

    return log_simplified
