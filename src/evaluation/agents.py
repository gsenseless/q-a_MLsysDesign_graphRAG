import json
import random
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from tqdm.auto import tqdm

from .logging import log_interaction_to_file
from .models import EvaluationChecklist, QuestionsList
from .prompts import get_evaluation_prompt, get_question_generation_prompt
from .rate_limiter import estimate_tokens, with_token_rate_limit


def setup_eval_agent() -> Agent:
    """Setup the evaluation agent."""
    evaluation_prompt = get_evaluation_prompt()
    eval_agent = Agent(
        name="eval_agent",
        model="mistral:mistral-small-latest",
        instructions=evaluation_prompt,
        output_type=EvaluationChecklist,
    )
    return eval_agent


def setup_question_generator() -> Agent:
    """Setup the question generator agent."""
    question_generation_prompt = get_question_generation_prompt()
    question_generator = Agent(
        name="question_generator",
        instructions=question_generation_prompt,
        model="mistral:mistral-small-latest",
        output_type=QuestionsList,
    )
    return question_generator


def setup_agents() -> tuple[Agent, Agent]:
    """Setup both eval and question generator agents."""
    return setup_eval_agent(), setup_question_generator()


async def generate_test_questions(
    question_generator: Agent,
    repo_data: list[dict[str, Any]],
    num_samples: int = 10,
) -> list[str]:
    """Generate test questions from data samples."""
    num_samples = min(num_samples, len(repo_data))
    sample = random.sample(repo_data, num_samples)

    questions = []
    batch_size = 5

    for i in tqdm(range(0, len(sample), batch_size), desc="Generating questions"):
        batch = sample[i : i + batch_size]
        prompt_docs = [d["content"] for d in batch]
        prompt = json.dumps(prompt_docs)

        result = await with_token_rate_limit(
            question_generator.run,
            prompt,
            estimated_input_tokens=estimate_tokens(prompt),
            estimated_output_tokens=300,
        )
        questions.extend(result.output.questions)

    return questions


async def run_agent_on_questions(
    agent: Agent,
    questions: list[str],
    log_dir: Path | str,
) -> None:
    """Run the agent on generated questions and log interactions."""
    log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir

    for q in tqdm(questions):
        print(q)

        try:
            result = await with_token_rate_limit(
                agent.run,
                user_prompt=q,
                estimated_input_tokens=estimate_tokens(q) + 500,
                estimated_output_tokens=500,
            )
            print(result.output[:200])  # Truncate output to save display space

            log_interaction_to_file(
                agent, result.new_messages(), log_dir, source="ai-generated"
            )
        except Exception as e:
            print(f"Error running agent on question '{q}': {e}")
            continue

        print()
