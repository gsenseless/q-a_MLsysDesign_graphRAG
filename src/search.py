import logging
import os
from pathlib import Path

import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from chunking import process_repo_chunks
from get_repo_data import read_repo_data

load_dotenv()

# Neo4j Settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,  # Redirects from stderr to stdout
)


class Neo4jGraphIndex:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.verify_connection()

    def verify_connection(self):
        import time

        max_retries = 5
        for i in range(max_retries):
            try:
                self.driver.verify_connectivity()
                logging.info("Connected to Neo4j")
                return
            except Exception as e:
                if i < max_retries - 1:
                    logging.warning(
                        f"Failed to connect to Neo4j (attempt {i + 1}/{max_retries}), retrying in 5s... {e}"
                    )
                    time.sleep(5)
                else:
                    logging.error(
                        f"Failed to connect to Neo4j after {max_retries} attempts: {e}"
                    )
                    raise e

    def close(self):
        self.driver.close()

    def query(self, cypher, params=None):
        logging.info(f"Executing Cypher: {cypher}")
        #  if params:
        #      logging.info(f"With params: {params}")
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            data = [record.data() for record in result]
            logging.info(f"Query returned {len(data)} rows")
            return data

    def create_constraints(self):
        # Create uniqueness constraints
        self.query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Folder) REQUIRE f.path IS UNIQUE"
        )
        self.query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE"
        )

    def create_vector_index(self, embedding_dim=768):
        # Create vector index on Chunk embedding
        self.query(f"""
        CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {embedding_dim},
            `vector.similarity_function`: 'cosine'
        }}}}
        """)
        # Create vector index on Folder embedding
        self.query(f"""
        CREATE VECTOR INDEX folder_vector_index IF NOT EXISTS
        FOR (f:Folder) ON (f.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {embedding_dim},
            `vector.similarity_function`: 'cosine'
        }}}}
        """)
        # Create vector index on File embedding
        self.query(f"""
        CREATE VECTOR INDEX file_vector_index IF NOT EXISTS
        FOR (f:File) ON (f.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {embedding_dim},
            `vector.similarity_function`: 'cosine'
        }}}}
        """)

    def insert_chunks(self, chunks):
        # Batch insert chunks using UNWIND
        cypher = """
        UNWIND $batch AS chunk
        MERGE (f:Folder {path: chunk.folder_path})
        ON CREATE SET f.name = chunk.folder_name
        MERGE (d:File {path: chunk.full_path})
        ON CREATE SET d.name = chunk.file_name
        MERGE (f)-[:CONTAINS]->(d)
        CREATE (c:Chunk {text: chunk.chunk})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        processed_batch = []
        for c in chunks:
            path = Path(c["filename"])
            folder_path = str(path.parent)
            c_copy = c.copy()
            c_copy["folder_path"] = folder_path
            processed_batch.append(c_copy)

        batch_size = 100
        for i in range(0, len(processed_batch), batch_size):
            batch = processed_batch[i : i + batch_size]
            self.query(cypher, {"batch": batch})

    def search_vector(self, query_embedding, num_results=5):
        # 1. Search for relevant folders
        # 2. Search for relevant files
        # 3. Search for relevant chunks (with higher limit)
        # 4. Boost chunks that are in relevant folders and files

        cypher = """
        CALL db.index.vector.queryNodes('folder_vector_index', 10, $embedding)
        YIELD node as folder, score as folder_score
        WITH collect({path: folder.path, score: folder_score}) as relevant_folders
        
        CALL db.index.vector.queryNodes('file_vector_index', 10, $embedding)
        YIELD node as file, score as file_score
        WITH relevant_folders, collect({path: file.path, score: file_score}) as relevant_files

        CALL db.index.vector.queryNodes('chunk_vector_index', $candidate_limit, $embedding)
        YIELD node as chunk, score as chunk_score
        
        MATCH (chunk)<-[:HAS_CHUNK]-(file)<-[:CONTAINS]-(folder)
        
        WITH chunk, chunk_score, file, folder, relevant_folders, relevant_files
        
        WITH chunk, chunk_score, file, folder, 
             [x IN relevant_folders WHERE x.path = folder.path | x.score] as f_scores,
             [y IN relevant_files WHERE y.path = file.path | y.score] as file_scores
             
        WITH chunk, chunk_score, file, folder,
             CASE WHEN size(f_scores) > 0 THEN head(f_scores) ELSE 0.0 END as folder_bonus,
             CASE WHEN size(file_scores) > 0 THEN head(file_scores) ELSE 0.0 END as file_bonus
             
        RETURN chunk.text as chunk, 
               chunk_score, 
               file.name as filename, 
               folder.name as folder, 
               folder_bonus,
               file_bonus,
               chunk_score + (folder_bonus * 0.5) + (file_bonus * 0.5) as score
        ORDER BY score DESC
        LIMIT $limit
        """
        candidate_limit = num_results * 10
        results = self.query(
            cypher,
            {
                "embedding": query_embedding,
                "limit": num_results,
                "candidate_limit": candidate_limit,
            },
        )
        logging.info(f"Vector search found {len(results)} results")
        if results:
            logging.info(f"Top result score: {results[0]['score']}")
        return results

    def add_embeddings(self, chunks, embedding_model):
        for i in range(0, len(chunks), 100):
            batch = chunks[i : i + 100]
            texts = [c["chunk"] for c in batch]
            embeddings = embedding_model.encode(texts)

            # Encode unique folder names in this batch
            folder_names = [c.get("folder_name", "root") for c in batch]
            unique_folders = list(set(folder_names))
            folder_emb_map = dict(
                zip(unique_folders, embedding_model.encode(unique_folders))
            )

            # Encode unique file names in this batch
            file_names = [c.get("file_name", "unknown") for c in batch]
            unique_files = list(set(file_names))
            file_emb_map = dict(zip(unique_files, embedding_model.encode(unique_files)))

            update_data = []
            for j, text in enumerate(texts):
                p = Path(batch[j].get("filename", "unknown"))
                folder_name = batch[j].get("folder_name", "root")
                file_name = batch[j].get("file_name", "unknown")

                update_data.append(
                    {
                        "text": text,
                        "embedding": embeddings[j].tolist(),
                        "folder_name": folder_name,
                        "folder_embedding": folder_emb_map[folder_name].tolist(),
                        "folder_path": str(p.parent),
                        "file_name": file_name,
                        "file_embedding": file_emb_map[file_name].tolist(),
                        "full_path": batch[j].get("full_path", "unknown"),
                    }
                )

            cypher = """
            UNWIND $batch AS item
            MERGE (f:Folder {path: item.folder_path})
            ON CREATE SET f.name = item.folder_name, f.embedding = item.folder_embedding
            ON MATCH SET f.embedding = item.folder_embedding
            MERGE (d:File {path: item.full_path})
            ON CREATE SET d.name = item.file_name, d.embedding = item.file_embedding
            ON MATCH SET d.embedding = item.file_embedding
            MERGE (f)-[:CONTAINS]->(d)
            MERGE (c:Chunk {text: item.text})
            ON CREATE SET c.embedding = item.embedding
            ON MATCH SET c.embedding = item.embedding
            MERGE (d)-[:HAS_CHUNK]->(c)
            """
            self.query(cypher, {"batch": update_data})


def create_vector_index(chunks):
    embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    embedding_dim = embedding_model.get_sentence_embedding_dimension()

    index = Neo4jGraphIndex(NEO4J_URI, (NEO4J_USERNAME, NEO4J_PASSWORD))
    index.create_constraints()
    index.create_vector_index(embedding_dim)
    index.add_embeddings(chunks, embedding_model)
    return index


def vector_search(query, docs_vindex, embedding_model):
    q_embedding = embedding_model.encode(query).tolist()
    return docs_vindex.search_vector(q_embedding)


if __name__ == "__main__":
    ml_system_design_repo = read_repo_data("ML-SystemDesign", "MLSystemDesign")
    print(f"Repo length: {len(ml_system_design_repo)}")

    ml_system_design_chunks = process_repo_chunks(
        ml_system_design_repo, "sliding_window"
    )
    print(f"Chunks length: {len(ml_system_design_chunks)}")

    embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    docs_vindex = create_vector_index(ml_system_design_chunks)

    query = "list essential sections of ml system design doc?"
    results = vector_search(query, docs_vindex, embedding_model)
    print(f"Vector Search Results: {len(results)}")
    for r in results:
        print(f"Score: {r['score']:.4f} - {r['filename']}: {r['chunk'][:100]}...")
