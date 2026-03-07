import json
import os
import random
from pathlib import Path

import pandas as pd
from pydantic_ai import Agent
from tqdm.auto import tqdm

from .agents import setup_agents
from .logging import load_log_file, simplify_log_messages
from .models import EvaluationChecklist
from .rate_limiter import (
    estimate_tokens,
    with_token_rate_limit,
)


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

    # Note: user_prompt already contains log, so just estimate the full prompt
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

        checks = {c.check_name: c.check_pass for c in eval_result.checklist}
        row.update(checks)

        rows.append(row)

    df_evals = pd.DataFrame(rows)
    return df_evals


async def generate_logs(log_dir: Path | str) -> None:
    """Generate test logs by running the agent on generated questions."""
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer

    from agent import create_repo_agent
    from chunking import process_repo_chunks
    from get_repo_data import read_repo_data
    from search import create_vector_index

    log_dir = Path(log_dir)
    if log_dir.exists():
        for f in log_dir.glob("*.json"):
            f.unlink()
    log_dir.mkdir(exist_ok=True)

    load_dotenv(".env")

    ml_system_design_repo = read_repo_data("ML-SystemDesign", "MLSystemDesign")
    print(len(ml_system_design_repo))

    ml_system_design_chunks = process_repo_chunks(
        ml_system_design_repo, "sliding_window"
    )
    print(len(ml_system_design_chunks))

    embedding_model = SentenceTransformer(
        os.getenv("EMBEDDING_MODEL_NAME", "multi-qa-distilbert-cos-v1")
    )
    docs_vindex = create_vector_index(ml_system_design_chunks, embedding_model)

    agent = create_repo_agent(docs_vindex, embedding_model)

    _, question_generator = setup_agents()

    from .agents import generate_test_questions, run_agent_on_questions

    questions = await generate_test_questions(
        question_generator, ml_system_design_repo, num_samples=30
    )
    questions = random.sample(questions, min(len(questions), 100))

    # Run agent on questions
    await run_agent_on_questions(agent, questions, log_dir)


async def evaluate_existing_logs(log_dir: Path | str) -> None:
    """Evaluate existing log files and generate a report."""
    from dotenv import load_dotenv

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

    # Create results dataframe
    df_evals = create_results_dataframe(eval_results)

    # Generate enhanced report
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
