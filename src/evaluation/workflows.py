import json
import os
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from agent import create_repo_agent
from chunking import process_repo_chunks
from get_repo_data import read_repo_data
from search import create_vector_index

from .agents import generate_test_questions, run_agent_on_questions, setup_agents
from .logging import load_log_file, simplify_log_messages
from .models import EvaluationChecklist
from .rate_limiter import estimate_tokens, with_token_rate_limit

_METRICS = (
    "factually_grounded",
    "key_information_retrieved",
    "search_relevance",
    "citation_accuracy",
    "formatting_compliance",
)


def _build_repo_agent():
    """Load repo data, build chunks and vector index, return (agent, repo_data)."""
    repo_data = read_repo_data("ML-SystemDesign", "MLSystemDesign")
    print(len(repo_data))

    chunks = process_repo_chunks(repo_data, "sliding_window")
    print(len(chunks))

    embedding_model = SentenceTransformer(
        os.getenv("EMBEDDING_MODEL_NAME", "multi-qa-distilbert-cos-v1")
    )
    docs_vindex = create_vector_index(chunks, embedding_model)
    agent = create_repo_agent(docs_vindex, embedding_model)
    return agent, repo_data


def extract_question_from_messages(messages: list) -> str:
    """Extract the user question from agent messages."""
    for m in messages:
        for part in m["parts"]:
            if part["part_kind"] == "user-prompt":
                return part["content"]
    return "Unknown Question"


async def evaluate_log_record(
    eval_agent: Agent,
    log_record: dict,
    user_prompt_format: str,
) -> EvaluationChecklist:
    """Evaluate a single log record."""
    messages = log_record["messages"]

    instructions = log_record["system_prompt"]
    question = extract_question_from_messages(messages)
    answer = messages[-1]["parts"][0]["content"]

    log_simplified = simplify_log_messages(messages)
    log = json.dumps(log_simplified)

    user_prompt = user_prompt_format.format(
        instructions=instructions, question=question, answer=answer, log=log
    )

    estimated_input = estimate_tokens(user_prompt)

    result = await with_token_rate_limit(
        eval_agent.run,
        user_prompt,
        output_type=EvaluationChecklist,
        estimated_input_tokens=estimated_input,
        estimated_output_tokens=500,
    )
    return result.output


def load_evaluation_set(log_dir: Path | str) -> list[dict]:
    """Load evaluation set from log files."""
    log_dir = Path(log_dir)
    eval_set = []
    for log_file in log_dir.glob("*.json"):
        log_record = load_log_file(log_file)
        if log_record["source"] != "ai-generated":
            continue
        eval_set.append(log_record)
    return eval_set


async def evaluate_logs(
    eval_agent: Agent,
    eval_set: list[dict],
    user_prompt_format: str,
) -> list[tuple[dict, EvaluationChecklist]]:
    """Evaluate all logs in the evaluation set."""
    eval_results = []
    for log_record in tqdm(eval_set):
        try:
            eval_result = await evaluate_log_record(
                eval_agent, log_record, user_prompt_format
            )
            eval_results.append((log_record, eval_result))
        except Exception as e:
            print(f"Error evaluating log {log_record.get('log_file')}: {e}")
            continue
    return eval_results


def create_results_dataframe(eval_results: list) -> pd.DataFrame:
    """Create a DataFrame from evaluation results."""
    rows = []
    for log_record, eval_result in eval_results:
        messages = log_record["messages"]
        question = extract_question_from_messages(messages)

        row = {
            "file": log_record["log_file"].name,
            "question": question,
            "answer": messages[-1]["parts"][0]["content"],
        }
        row.update({metric: getattr(eval_result, metric).passed for metric in _METRICS})
        rows.append(row)

    return pd.DataFrame(rows)


async def generate_logs(log_dir: Path | str) -> None:
    """Phase 1: Run the repo agent on generated questions and save interaction logs."""
    log_dir = Path(log_dir)
    if log_dir.exists():
        for f in log_dir.glob("*.json"):
            f.unlink()
    log_dir.mkdir(exist_ok=True)

    load_dotenv(".env")

    agent, repo_data = _build_repo_agent()

    _, question_generator = setup_agents()
    questions = await generate_test_questions(question_generator, repo_data, num_samples=30)
    questions = random.sample(questions, min(len(questions), 100))

    await run_agent_on_questions(agent, questions, log_dir)


async def evaluate_existing_logs(log_dir: Path | str) -> None:
    """Phase 2: Score saved logs with an LLM-as-a-Judge and print the report."""
    log_dir = Path(log_dir)
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

    df_evals = create_results_dataframe(eval_results)

    mean_scores = df_evals.mean(numeric_only=True)
    report_df = pd.DataFrame(
        {
            "Metric": mean_scores.index,
            "Score": (mean_scores.values * 100).round(1).astype(str) + "%",
        }
    )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT")
    print(f"Total Questions Evaluated: {len(df_evals)}")
    print("-" * 60)
    print(report_df.to_string(index=False))
    print("=" * 60 + "\n")
