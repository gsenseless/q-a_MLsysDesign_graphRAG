import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from get_repo_data import read_repo_data
from chunking import process_repo_chunks
from search import create_vector_index
from agent import create_repo_agent

load_dotenv(".env")

# Import evaluation modules
import random
import json
import pandas as pd
from pathlib import Path
from eval import (
    setup_eval_agent,
    setup_question_generator,
    log_entry,
    evaluate_log_record,
    get_evaluation_prompt,
    simplify_log_messages,
    serializer,
    create_results_dataframe,
)

# Page config
st.set_page_config(page_title="ML System Design Repository", layout="wide")

st.title("ML system design repository Q&A")
st.markdown(
    "[ML-SystemDesign Repository](https://github.com/ML-SystemDesign/MLSystemDesign)"
)


@st.cache_resource
def initialize_resources():
    """
    Initialize the resources, loading data and creating the vector index.
    This is cached so it runs only once per session/runtime.
    """
    with st.status(
        "Initializing agent and processing repository...", expanded=True
    ) as status:
        st.write("Reading repository data...")
        ml_system_design_repo = read_repo_data("ML-SystemDesign", "MLSystemDesign")

        st.write("Chunking repository contents...")
        ml_system_design_chunks = process_repo_chunks(
            ml_system_design_repo, "sliding_window"
        )

        st.write("Loading embedding model...")
        embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")

        st.write("Creating vector index...")
        docs_vindex = create_vector_index(ml_system_design_chunks)

        return docs_vindex, embedding_model, ml_system_design_repo


# This will display the spinner on first run
docs_vindex, embedding_model, repo_data = initialize_resources()

# Create Tabs
tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

with tab_chat:
    question = st.text_input("Ask a question about ML System Design:")

    if st.button("Run"):
        if question:
            with st.spinner("Thinking..."):
                # Run the agent
                # run_sync is used because we are in a synchronous Streamlit context
                try:
                    # Create a fresh agent for each run to avoid loop issues
                    agent = create_repo_agent(docs_vindex, embedding_model)
                    # Use run_sync instead of asyncio.run(agent.run) to avoid event loop issues in Streamlit
                    result = agent.run_sync(user_prompt=question)

                    st.markdown("### Answer")

                    messages = []
                    final_output = None

                    if hasattr(result, "new_messages"):
                        messages = result.new_messages()
                        final_output = getattr(
                            result, "output", getattr(result, "data", None)
                        )
                    elif isinstance(result, list):
                        messages = result

                    if final_output:
                        st.write(str(final_output))
                    else:
                        st.error("The agent failed to generate a final answer.")
                        st.info("""
                        **Reload the page. Restart the app**
                        """)

                    with st.expander("Sources"):
                        found_chunks = False
                        for message in messages:
                            parts = getattr(message, "parts", [])
                            for part in parts:
                                kind = getattr(part, "part_kind", "unknown")
                                if kind == "tool-return":
                                    content = getattr(part, "content", None)
                                    if isinstance(content, list):
                                        found_chunks = True
                                        for item in content:
                                            if isinstance(item, dict):
                                                folder = item.get("folder", "unknown")
                                                filename = item.get(
                                                    "filename", "unknown"
                                                )
                                                st.markdown(
                                                    f"<span style='color:gray'>{folder} > {filename}</span>",
                                                    unsafe_allow_html=True,
                                                )

                        if not found_chunks:
                            st.write("No specific sources used.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")

with tab_eval:
    st.header("Agent Evaluation Benchmark")
    st.write("Generate questions, run the agent, and evaluate the quality of answers.")

    if st.button("Start Evaluation"):
        # Progress container
        status_container = st.status("Starting evaluation...", expanded=True)

        try:
            # 1. Setup Agents
            status_container.write("Initializing evaluation agents...")
            eval_agent = setup_eval_agent()
            question_generator = setup_question_generator()

            # 2. Generate Questions
            status_container.write("Generating test questions from repository data...")
            # We use run_sync manually here to match the sync context
            sample = random.sample(repo_data, 5)  # Generate 2 questions as per default
            prompt_docs = [d["content"] for d in sample]
            prompt_json = json.dumps(prompt_docs)

            q_gen_result = question_generator.run_sync(prompt_json)
            questions = q_gen_result.output.questions

            st.write(f"**Generated Questions:**")
            st.json(questions)

            # 3. Run Agent and Evaluate
            status_container.write("Running agent on questions and evaluating...")

            eval_results = []

            eval_progress = st.progress(0.0)

            user_prompt_format = """
                <INSTRUCTIONS>{instructions}</INSTRUCTIONS>
                <QUESTION>{question}</QUESTION>
                <ANSWER>{answer}</ANSWER>
                <LOG>{log}</LOG>
            """.strip()

            for i, q in enumerate(questions):
                status_container.write(
                    f"Processing question {i + 1}/{len(questions)}: {q}"
                )

                # a. Run Agent
                repo_agent = create_repo_agent(docs_vindex, embedding_model)
                run_result = repo_agent.run_sync(user_prompt=q)

                # b. Log
                new_msgs = run_result.new_messages()
                log_record = log_entry(repo_agent, new_msgs, source="ai-generated")
                log_record["log_file"] = Path(f"question_{i + 1}.json")

                messages = log_record["messages"]
                instructions = log_record["system_prompt"]
                question_text = "Unknown Question"
                for m in messages:
                    for part in m["parts"]:
                        if part["part_kind"] == "user-prompt":
                            question_text = part["content"]
                            break
                    if question_text != "Unknown Question":
                        break
                answer_text = messages[-1]["parts"][0]["content"]

                log_simplified = simplify_log_messages(messages)
                log_str = json.dumps(log_simplified, default=serializer)

                eval_prompt = user_prompt_format.format(
                    instructions=instructions,
                    question=question_text,
                    answer=answer_text,
                    log=log_str,
                )

                eval_result_obj = eval_agent.run_sync(eval_prompt)
                eval_result = eval_result_obj.output

                eval_results.append((log_record, eval_result))
                eval_progress.progress((i + 1) / len(questions))

            status_container.update(
                label="Evaluation Complete!", state="complete", expanded=False
            )

            # 4. Display Results
            st.subheader("Evaluation Results")
            df_evals = create_results_dataframe(eval_results)
            st.dataframe(df_evals)

            mean_scores = df_evals.select_dtypes(include=["bool", "number"]).mean()
            st.write("**Average Passing Rates:**")
            st.dataframe(
                mean_scores.map(lambda x: f"{x:.1%}"), use_container_width=True
            )

        except Exception as e:
            st.error(f"An error occurred during evaluation: {e}")
            import traceback

            st.code(traceback.format_exc())
