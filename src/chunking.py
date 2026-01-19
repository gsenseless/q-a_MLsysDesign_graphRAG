import re
from pathlib import Path


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i : i + size]
        result.append({"start": i, "chunk": chunk})
        if i + size >= n:
            break

    return result


def split_markdown_by_level(text, level=2):
    """
    Split markdown text by a specific header level.

    :param text: Markdown text as a string
    :param level: Header level to split on
    :return: List of sections as strings
    """
    # This regex matches markdown headers
    # For level 2, it matches lines starting with "## "
    header_pattern = r"^(#{" + str(level) + r"} )(.+)$"
    pattern = re.compile(header_pattern, re.MULTILINE)

    # Split and keep the headers
    parts = pattern.split(text)

    sections = []
    for i in range(1, len(parts), 3):
        # We step by 3 because regex.split() with
        # capturing groups returns:
        # [before_match, group1, group2, after_match, ...]
        # here group1 is "## ", group2 is the header text
        header = parts[i] + parts[i + 1]  # "## " + "Title"
        header = header.strip()

        # Get the content after this header
        content = ""
        if i + 2 < len(parts):
            content = parts[i + 2].strip()

        section = f"{header}\n\n{content}" if content else header
        sections.append(section)

    return sections


def process_repo_chunks(repo, chunking_method_name):
    """
    Processes a repository of documents into chunks using a specified chunking method.
    extracts metadata for GraphRAG (Folder -> File -> Chunk).

    :param repo: A list of document dictionaries, each containing at least a 'content' key.
    :param chunking_method_name: A string indicating the chunking method to use ('sliding_window' or 'split_markdown_by_level').
    :return: A list of chunked documents.
    """
    all_chunks = []
    for doc in repo:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content")
        filename = doc_copy.get("filename", "")

        # Structure Extraction
        path = Path(filename)
        # Using immediate parent as the folder.
        # If the structure is strictly Folder/File, this captures the Folder.
        # If deeply nested, it captures the containing folder.
        parent = path.parent
        if str(parent) == ".":
            folder_name = "root"
        else:
            folder_name = parent.name

        file_name = path.name

        doc_copy["folder_name"] = folder_name
        doc_copy["file_name"] = file_name
        doc_copy["full_path"] = str(path)

        if chunking_method_name == "sliding_window":
            chunks = sliding_window(doc_content, 2000, 1000)
            for chunk in chunks:
                chunk_data = doc_copy.copy()
                chunk_data.update(chunk)
                all_chunks.append(chunk_data)
        elif chunking_method_name == "split_markdown_by_level":
            sections = split_markdown_by_level(doc_content, level=2)
            for section_content in sections:
                chunk_data = doc_copy.copy()
                chunk_data["chunk"] = section_content
                all_chunks.append(chunk_data)
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method_name}")
    return all_chunks


if __name__ == "__main__":
    from get_repo_data import read_repo_data

    ml_system_design_repo = read_repo_data("ML-SystemDesign", "MLSystemDesign")
    print(len(ml_system_design_repo))

    ml_system_design_chunks = process_repo_chunks(
        ml_system_design_repo, "sliding_window"
    )
    print(len(ml_system_design_chunks))
    if len(ml_system_design_chunks) > 0:
        print(ml_system_design_chunks[0])
