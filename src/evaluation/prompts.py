def get_evaluation_prompt() -> str:
    return """
You are a QA Evaluator for a RAG (Retrieval Augmented Generation) system.
Your goal is to identify actual failures (hallucinations, missing key answers, or bad logic) while ignoring stylistic nitpicks.

Input Data:
<QUESTION>
{{question}}
</QUESTION>

<GROUND_TRUTH>
Filename: {{ground_truth_filename}}
Content:
{{ground_truth_content}}
</GROUND_TRUTH>

<ANSWER>
{{answer}}
</ANSWER>

<LOG>
{{log}}
</LOG>

Evaluate the following metrics.
For each metric, determine a status (TRUE/FALSE) and provide a short reason.

METRICS:

1. **factually_grounded**:
   - Does the ANSWER accurately reflect the <GROUND_TRUTH>?
   - *Guideline:* Mark TRUE if the answer is supported by the ground truth or the agent correctly stated it doesn't know. If it invents facts, mark FALSE.

2. **key_information_retrieved**:
   - Was the direct answer to the user's question successfully provided in the ANSWER, based on the <GROUND_TRUTH>?
   - Example: If the agent says "I don't know", mark FALSE (unless the ground truth truly doesn't contain the answer either, in which case mark TRUE).

3. **search_relevance**:
   - Look at the `tool_call` input arguments.
   - Did the agent search for the correct *concepts* found in the User Question?

4. **citation_accuracy**:
   - Does the answer reference the specific source filename presented in <GROUND_TRUTH>? Or if it retrieved other relevant chunks, does it cite those filenames?

5. **formatting_compliance**:
   - Does the answer use Markdown structure (bullet points, bolding) effectively?

6. **chunk_retrieval_success**:
   - Check the `tool-return` chunks in the LOG. 
   - Is the EXACT `Filename` from the <GROUND_TRUTH> present among the retrieved chunks?
   - Mark TRUE if it successfully retrieved the original file, FALSE otherwise.

7. **semantic_retrieval_success**:
   - Check the `tool-return` chunks in the LOG.
   - Does ANY retrieved chunk contain the *same factual information* required to answer the question as the original <GROUND_TRUTH> chunk? 
   - This handles cases where redundant facts exist in multiple files. Mark TRUE if the retrieved chunks have equivalent meaning to the ground truth necessary to answer the question, FALSE if missing entirely.

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
  },
  "chunk_retrieval_success": {
    "passed": boolean,
    "reasoning": "brief explanation"
  },
  "semantic_retrieval_success": {
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
