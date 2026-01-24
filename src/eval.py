import json
import random
import secrets
from datetime import datetime
from pathlib import Path

import pandas as pd
import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

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
    question = "Unknown Question"
    for m in messages:
        for part in m["parts"]:
            if part["part_kind"] == "user-prompt":
                question = part["content"]
                break
        if question != "Unknown Question":
            break
    answer = messages[-1]["parts"][0]["content"]

    log_simplified = simplify_log_messages(messages)
    log = json.dumps(log_simplified)

    user_prompt = user_prompt_format.format(
        instructions=instructions, question=question, answer=answer, log=log
    )

    result = await eval_agent.run(user_prompt, output_type=EvaluationChecklist)
    return result.output


def get_evaluation_prompt():
    ### reasoning field in JSON is not used but it should help the model to think deeper.
    return """
You are a QA Evaluator for a RAG (Retrieval Augmented Generation) system.
Your goal is to identify actual failures (hallucinations, missing key answers, or bad logic) while ignoring stylistic nitpicks.

Input Data:
<QUESTION>
{{question}}
</QUESTION>

<ANSWER>
{{answer}}
</ANSWER>

<LOG>
{{log}}
</LOG>

Evaluate the following 4 metrics. 
For each metric, determine a status (TRUE/FALSE) and provide a short reason.

METRICS:

1. **factually_grounded**: 
   - Check the `tool-return` chunks in the LOG. 
   - Does the ANSWER contradict the information in the chunks?
   - Does the ANSWER contain specific data (dates, names, types) that is NOT present in the chunks? (General knowledge is okay, specific fabricated data is a failure).
   - *Guideline:* If the answer is supported by the context or reasonably inferred from it, mark TRUE. If it invents facts, mark FALSE.

2. **key_information_retrieved**:
   - Looking at the chunks, was there a direct answer to the user's question that the Agent missed?
   - Example: If the text lists "5 types of fruit" and the Agent says "I don't know" or only lists 1, mark FALSE.
   - Example: If the text lists "5 types of fruit" and the Agent lists all 5 or summarizes them accurately, mark TRUE.

3. **search_relevance**:
   - Look at the `tool_call` input arguments.
   - Did the agent search for the correct *concepts* found in the User Question?
   - *Guideline:* Mark FALSE only if the search query was totally irrelevant or if the agent failed to search when it clearly needed to.

4. **citation_accuracy**:
   - Does the answer reference the specific source filename (found in the 'filename' field of the chunks, e.g., 'CONTRIBUTING.md')? 
   - General phrases like "the repository" or "the context" are insufficient. It must cite the specific document name.

5. **formatting_compliance**:
   - Does the answer use Markdown structure (bullet points, bolding) effectively to match the structure of the retrieved data?

Output Format (JSON):
{
  "factually_grounded": {
    "passed": boolean,
    "reasoning": "..."
  },
  "key_information_retrieved": {
    "passed": boolean,
    "reasoning": "..."
  },
  "search_relevance": {
    "passed": boolean,
    "reasoning": "..."
  },
  "citation_accuracy": {
    "passed": boolean,
    "reasoning": "brief explanation"
  },
  "formatting_compliance": {
    "passed": boolean,
    "reasoning": "brief explanation"
  }
}
""".strip()


def get_question_generation_prompt():
    return """
You are helping to create test questions for an AI agent that answers questions about ML system design.

Based on the provided content, generate realistic questions that readers might ask.

The questions should:

- Be varied in style
- Range from simple to complex
- Include both specific technical questions and general questions

""".strip()


def setup_eval_agent():
    evaluation_prompt = get_evaluation_prompt()
    eval_agent = Agent(
        name="eval_agent",
        model="mistral:mistral-small-latest",
        instructions=evaluation_prompt,
        output_type=EvaluationChecklist,
    )
    return eval_agent


def setup_question_generator():
    question_generation_prompt = get_question_generation_prompt()
    question_generator = Agent(
        name="question_generator",
        instructions=question_generation_prompt,
        model="mistral:mistral-small-latest",
        output_type=QuestionsList,
    )
    return question_generator


def setup_agents():
    return setup_eval_agent(), setup_question_generator()


async def generate_test_questions(question_generator, repo_data, num_samples=10):
    """Generate test questions from data samples."""
    num_samples = min(num_samples, len(repo_data))
    sample = random.sample(repo_data, num_samples)

    questions = []
    batch_size = 5

    for i in tqdm(range(0, len(sample), batch_size), desc="Generating questions"):
        batch = sample[i : i + batch_size]
        prompt_docs = [d["content"] for d in batch]
        prompt = json.dumps(prompt_docs)

        result = await question_generator.run(prompt)
        questions.extend(result.output.questions)

    return questions


async def run_agent_on_questions(agent, questions, log_dir):
    """Run the agent on generated questions and log interactions."""
    for q in tqdm(questions):
        print(q)

        try:
            result = await agent.run(user_prompt=q)
            print(result.output)

            log_interaction_to_file(
                agent, result.new_messages(), log_dir, source="ai-generated"
            )
        except Exception as e:
            print(f"Error running agent on question '{q}': {e}")
            continue

        print()


def load_evaluation_set(LOG_DIR):
    """Load evaluation set from log files."""
    eval_set = []
    for log_file in LOG_DIR.glob("*.json"):
        
        log_record = load_log_file(log_file)
        if log_record["source"] != "ai-generated":
            continue

        eval_set.append(log_record)

    return eval_set


async def evaluate_logs(eval_agent, eval_set, user_prompt_format):
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


def create_results_dataframe(eval_results):
    """Create a DataFrame from evaluation results."""
    rows = []
    for log_record, eval_result in eval_results:
        messages = log_record["messages"]

        # Find the user question
        question = "Unknown Question"
        for m in messages:
            for part in m["parts"]:
                if part["part_kind"] == "user-prompt":
                    question = part["content"]
                    break
            if question != "Unknown Question":
                break

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


async def generate_logs(log_dir):
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

    embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    docs_vindex = create_vector_index(ml_system_design_chunks)

    agent = create_repo_agent(docs_vindex, embedding_model)

    _, question_generator = setup_agents()

    questions = await generate_test_questions(
        question_generator, ml_system_design_repo, num_samples=30
    )
    questions = random.sample(questions, min(len(questions), 100))

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
    
    # Generate enhanced report
    mean_scores = df_evals.mean(numeric_only=True)
    report_df = pd.DataFrame({
        "Metric": mean_scores.index,
        "Score": (mean_scores.values * 100).round(1).astype(str) + "%"
    })
    
    print("\n" + "="*60)
    print(f"FINAL EVALUATION REPORT")
    print(f"Total Questions Evaluated: {len(df_evals)}")
    print("-" * 60)
    print(report_df.to_string(index=False))
    print("="*60 + "\n")


async def main():
    log_dir = Path("logs")
    await generate_logs(log_dir)
    await evaluate_existing_logs(log_dir)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
