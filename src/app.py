import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from get_repo_data import read_repo_data
from chunking import process_repo_chunks
from search import create_vector_index
from agent import create_repo_agent

load_dotenv(".env")

# Page config
st.set_page_config(page_title="ML System Design Repository", layout="wide")

st.title("ML system design repository Q&A")
st.markdown("[ML-SystemDesign Repository](https://github.com/ML-SystemDesign/MLSystemDesign)")

@st.cache_resource
def initialize_resources():
    """
    Initialize the resources, loading data and creating the vector index.
    This is cached so it runs only once per session/runtime.
    """
    with st.status("Initializing agent and processing repository...", expanded=True) as status:
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
        
        return docs_vindex, embedding_model

# This will display the spinner on first run
docs_vindex, embedding_model = initialize_resources()

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
                
                # --- Display Answer ---
                st.markdown("### Answer")
                
                messages = []
                final_output = None
                
                if hasattr(result, "new_messages"):
                    messages = result.new_messages()
                    final_output = getattr(result, "output", getattr(result, "data", None))
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
                                            filename = item.get("filename", "unknown")
                                            st.markdown(f"<span style='color:gray'>{folder} > {filename}</span>", unsafe_allow_html=True)
                    
                    if not found_chunks:
                        st.write("No specific sources used.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
