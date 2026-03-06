from pydantic import BaseModel
from pydantic_ai import Agent


class EvaluationCheck(BaseModel):
    check_name: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck]


class QuestionsList(BaseModel):
    questions: list[str]


def get_evaluation_prompt() -> str:
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


def get_question_generation_prompt() -> str:
    return """
You are helping to create test questions for an AI agent that answers questions about ML system design.

Based on the provided content, generate realistic questions that readers might ask.

The questions should:

- Be varied in style
- Range from simple to complex
- Include both specific technical questions and general questions
""".strip()
