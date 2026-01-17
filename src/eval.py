import pretty_errors
import json
import random
import secrets
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer
from agent import create_repo_agent
from chunking import process_repo_chunks
from get_repo_data import read_repo_data
from search import create_vector_index


class EvaluationCheck(BaseModel):
    check_name: str
    # justification: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck]
    # summary: str


class QuestionsList(BaseModel):
    questions: list[str]


def serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def log_entry(agent, messages, source="user"):
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


def log_interaction_to_file(agent, messages, log_dir, source="user"):
    entry = log_entry(agent, messages, source)

    ts = entry["messages"][-1]["timestamp"]
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
    filepath = log_dir / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    return filepath


def load_log_file(log_file):
    with Path(log_file).open() as f_in:
        log_data = json.load(f_in)
        log_data["log_file"] = log_file
        return log_data


def simplify_log_messages(messages):
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


async def evaluate_log_record(eval_agent, log_record, user_prompt_format):
    messages = log_record["messages"]

    instructions = log_record["system_prompt"]
    question = messages[0]["parts"][0]["content"]
    answer = messages[-1]["parts"][0]["content"]

    log_simplified = simplify_log_messages(messages)
    log = json.dumps(log_simplified)

    user_prompt = user_prompt_format.format(
        instructions=instructions, question=question, answer=answer, log=log
    )

    result = await eval_agent.run(user_prompt, output_type=EvaluationChecklist)
    return result.output


def get_evaluation_prompt():
    return """
Use this checklist to evaluate the quality of an AI agent's answer (<ANSWER>) to a user question (<QUESTION>).
We also include the entire log (<LOG>) for analysis.

For each item, check if the condition is met. 

Checklist:

- instructions_follow: The agent followed the user's instructions (in <INSTRUCTIONS>)
- instructions_avoid: The agent avoided doing things it was told not to do  
- answer_relevant: The response directly addresses the user's question  
- answer_clear: The answer is clear and correct  
- answer_citations: The response includes proper citations or sources when required  
- completeness: The response is complete and covers all key aspects of the request
- tool_call_search: Is the search tool invoked? 

Output true/false for each check and provide a short explanation for your judgment.
""".strip()


def get_question_generation_prompt():
    return """
You are helping to create test questions for an AI agent that answers questions about a data engineering course.

Based on the provided content, generate realistic questions that readers might ask.

The questions should:

- Be varied in style
- Range from simple to complex
- Include both specific technical questions and general questions

Generate one question for each record.
""".strip()


def setup_agents():
    evaluation_prompt = get_evaluation_prompt()
    eval_agent = Agent(
        name="eval_agent",
        model="mistral:mistral-small-latest",
        instructions=evaluation_prompt,
        output_type=EvaluationChecklist,
    )

    question_generation_prompt = get_question_generation_prompt()
    question_generator = Agent(
        name="question_generator",
        instructions=question_generation_prompt,
        model="mistral:mistral-small-latest",
        output_type=QuestionsList,
    )

    return eval_agent, question_generator


async def generate_test_questions(question_generator, repo_data, num_samples=2):
    """Generate test questions from data samples."""
    sample = random.sample(repo_data, num_samples)
    prompt_docs = [d["content"] for d in sample]
    prompt = json.dumps(prompt_docs)

    result = await question_generator.run(prompt)
    questions = result.output.questions

    return questions


async def run_agent_on_questions(agent, questions, log_dir):
    """Run the agent on generated questions and log interactions."""
    for q in tqdm(questions):
        print(q)

        result = await agent.run(user_prompt=q)
        print(result.output)

        log_interaction_to_file(
            agent, result.new_messages(), log_dir, source="ai-generated"
        )

        print()


def load_evaluation_set(LOG_DIR, agent_name="repo_agent"):
    """Load evaluation set from log files."""
    eval_set = []
    for log_file in LOG_DIR.glob("*.json"):
        if agent_name not in log_file.name:
            continue

        log_record = load_log_file(log_file)
        if log_record["source"] != "ai-generated":
            continue

        eval_set.append(log_record)

    return eval_set


async def evaluate_logs(eval_agent, eval_set, user_prompt_format):
    """Evaluate all logs in the evaluation set."""
    eval_results = []
    for log_record in tqdm(eval_set):
        eval_result = await evaluate_log_record(
            eval_agent, log_record, user_prompt_format
        )
        eval_results.append((log_record, eval_result))

    return eval_results


def create_results_dataframe(eval_results):
    """Create a DataFrame from evaluation results."""
    rows = []
    for log_record, eval_result in eval_results:
        messages = log_record["messages"]

        row = {
            "file": log_record["log_file"].name,
            "question": messages[0]["parts"][0]["content"],
            "answer": messages[-1]["parts"][0]["content"],
        }

        checks = {c.check_name: c.check_pass for c in eval_result.checklist}
        row.update(checks)

        rows.append(row)

    df_evals = pd.DataFrame(rows)
    return df_evals


async def generate_logs(log_dir):
    log_dir.mkdir(exist_ok=True)

    load_dotenv(".env")

    ml_system_design_repo = read_repo_data("ML-SystemDesign", "MLSystemDesign")
    print(len(ml_system_design_repo))

    ml_system_design_chunks = process_repo_chunks(
        ml_system_design_repo, "sliding_window"
    )
    print(len(ml_system_design_chunks))

    repo_data = ml_system_design_repo

    embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    docs_vindex = create_vector_index(ml_system_design_chunks)

    agent = create_repo_agent(docs_vindex, embedding_model)

    _, question_generator = setup_agents()

    questions = await generate_test_questions(
        question_generator, repo_data, num_samples=2
    )

    # Run agent on questions
    await run_agent_on_questions(agent, questions, log_dir)


async def evaluate_existing_logs(log_dir):
    load_dotenv(".env")

    user_prompt_format = """
        <INSTRUCTIONS>{instructions}</INSTRUCTIONS>
        <QUESTION>{question}</QUESTION>
        <ANSWER>{answer}</ANSWER>
        <LOG>{log}</LOG>
        """.strip()

    eval_agent, _ = setup_agents()

    eval_set = load_evaluation_set(log_dir)
    print(len(eval_set))

    eval_results = await evaluate_logs(eval_agent, eval_set, user_prompt_format)

    # Create results dataframe
    df_evals = create_results_dataframe(eval_results)
    print(df_evals.mean(numeric_only=True))


async def main():
    log_dir = Path("logs")
    # await generate_logs(log_dir)
    await evaluate_existing_logs(log_dir)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
