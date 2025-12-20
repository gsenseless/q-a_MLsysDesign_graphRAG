import numpy as np
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from chunking import process_repo_chunks
from get_repo_data import read_repo_data


def text_search(query, docs_index):
    return docs_index.search(query, num_results=10)


def vector_search(query, docs_vindex, embedding_model):
    q = embedding_model.encode(query)
    return docs_vindex.search(q, num_results=5)


def hybrid_search(query, docs_index, docs_vindex, embedding_model):
    text_results = text_search(query, docs_index)
    vector_results = vector_search(query, docs_vindex, embedding_model)

    # Combine and deduplicate results
    seen_ids = set()
    combined_results = []

    for result in text_results + vector_results:
        ### only single chunk per one file.
        if result["filename"] not in seen_ids:
            seen_ids.add(result["filename"])
            combined_results.append(result)

    return combined_results


def create_docs_index(chunks):
    docs_index = Index(
        text_fields=["chunk", "title", "description", "filename"], keyword_fields=[]
    )
    docs_index.fit(chunks)
    return docs_index


def create_vector_index(chunks):
    docs_embeddings = []

    for d in tqdm(chunks, desc="Generating embeddings"):
        text = d["chunk"]
        v = embedding_model.encode(text)
        docs_embeddings.append(v)

    docs_embeddings = np.array(docs_embeddings)

    docs_vindex = VectorSearch()
    docs_vindex.fit(docs_embeddings, chunks)
    return docs_vindex


if __name__ == "__main__":
    ml_system_design_repo = read_repo_data("ML-SystemDesign", "MLSystemDesign")
    print(len(ml_system_design_repo))

    ml_system_design_chunks = process_repo_chunks(
        ml_system_design_repo, "sliding_window"
    )
    print(len(ml_system_design_chunks))
    # ml_system_design_chunks = process_repo_chunks(ml_system_design_repo, 'split_markdown_by_level')
    # print(len(ml_system_design_chunks))

    # print(ml_system_design_chunks[1])

    ### -----

    docs_index = create_docs_index(ml_system_design_chunks)

    print("---------------")
    query = "What should be in a test dataset for AI evaluation?"
    results = text_search(query, docs_index)
    # results = docs_index.search(query)
    print(len(results))

    # ------------

    embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")

    docs_vindex = create_vector_index(ml_system_design_chunks)

    query = "typical chapters in ML sys design doc"
    results = vector_search(query, docs_vindex, embedding_model)
    # q = embedding_model.encode(query)
    # results = docs_vindex.search(q)
    for i, result in enumerate(results):
        print(result)
        if i < len(results) - 1:
            print("-----")
    print(len(results))

    # ---- hybrid:
    results = hybrid_search(query, docs_index, docs_vindex, embedding_model)
    for i, result in enumerate(results):
        print(result)
        if i < len(results) - 1:
            print("-----")
    print(len(results))
