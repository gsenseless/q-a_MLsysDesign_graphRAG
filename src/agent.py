import asyncio
import logging
import re
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent

from chunking import process_repo_chunks
from get_repo_data import read_repo_data
from search import create_docs_index


def create_repo_agent(docs_index):
    def text_search_tool(query: str) -> list[Any]:
        """
        Perform a text-based search on the data index.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A list of up to 5 search results returned by the data index.
        """
        from search import text_search

        return text_search(query, docs_index)

    system_prompt = """
    You are a helpful assistant for a ML system design repository. 

    Use the search tool to find relevant information from the repository before answering questions.

    If you can find specific information through search, use it to provide accurate answers.
    If the search doesn't return relevant results, let the user know and provide general guidance.
    """

    agent = Agent(
        name="repo_agent_v2",
        instructions=system_prompt,
        tools=[text_search_tool],
        model="mistral:mistral-small-latest",
    )

    return agent


def prettify_trace_log(trace: str) -> str:
    """
    Convert a raw LLM trace log into a readable, structured format.
    """

    output = []
    run_id_match = re.search(r"run_id='([^']+)'", trace)
    run_id = run_id_match.group(1) if run_id_match else "UNKNOWN"

    output.append(f"=== RUN {run_id} ===\n")

    # --- User prompt ---
    user_prompt_matches = re.findall(r"UserPromptPart\(content='([^']+)'", trace)
    for prompt in user_prompt_matches:
        output.append("[USER PROMPT]")
        output.append(prompt)
        output.append("")

    # --- Tool calls ---
    tool_calls = re.findall(r"ToolCallPart\(tool_name='([^']+)', args='([^']+)'", trace)
    for tool, args in tool_calls:
        output.append("[MODEL → TOOL CALL]")
        output.append(f"Tool: {tool}")
        output.append(f"Args: {args}")
        output.append("")

    # --- Tool returns ---
    tool_returns = re.findall(
        r"ToolReturnPart\(tool_name='([^']+)', content=\[(.*?)\], tool_call_id",
        trace,
        re.DOTALL,
    )

    for tool, content in tool_returns:
        output.append("[TOOL RESULT]")
        output.append(f"Tool: {tool}")

        chunks = re.findall(
            r"'chunk': '(.+?)', 'filename': '(.+?)'",
            content,
            re.DOTALL,
        )

        for chunk, filename in chunks[:2]:  # limit noise
            snippet = chunk.replace("\n", " ").strip()
            snippet = snippet[:300] + ("..." if len(snippet) > 300 else "")
            output.append(f"Source: {filename}")
            output.append(f"Snippet: {snippet}")

        output.append("")

    # --- Model responses ---
    responses = re.findall(
        r"TextPart\(content='(.+?)'\)\]",
        trace,
        re.DOTALL,
    )

    for response in responses:
        clean_response = response.replace("\\n", "\n").strip()
        output.append("[MODEL RESPONSE]")
        output.append(clean_response)
        output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    load_dotenv(".env")

    ml_system_design_repo = read_repo_data("ML-SystemDesign", "MLSystemDesign")
    print(len(ml_system_design_repo))

    ml_system_design_chunks = process_repo_chunks(
        ml_system_design_repo, "sliding_window"
    )
    print(len(ml_system_design_chunks))

    docs_index = create_docs_index(ml_system_design_chunks)

    ### -----
    agent = create_repo_agent(docs_index)

    question = "list essential sections of ml system design doc?"

    result = asyncio.run(agent.run(user_prompt=question))

    print(result)

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(result.new_messages())
    print(prettify_trace_log(str(result.new_messages())))
