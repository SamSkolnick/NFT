import chromadb

# Initialize the ChromaDB client. 
# 'PersistentClient' saves the data to disk in the specified path.
client = chromadb.PersistentClient(path="./agent_memory_db")

# Create a 'collection' which is like a table in a traditional database.
# If the collection already exists, it will load the existing one.
collection = client.get_or_create_collection(name="research_and_development")

def store_memory(doc_id: str, document: str, metadata: dict):
    """
    Stores a piece of text (a 'memory') in the vector database.
    'upsert' will add the document if it's new or update it if the id already exists.
    """
    collection.upsert(
        documents=[document],
        metadatas=[metadata],
        ids=[doc_id]
    )
    print(f"Memory stored with ID: {doc_id}")

def retrieve_memories(query_text: str, n_results: int = 2) -> list:
    """
    Queries the database to find the most relevant memories based on the query text.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results['documents'][0]

# --- EXAMPLE USAGE ---

# Let's imagine your agent just processed a research paper.
# It would extract key concepts and store them as memories.
store_memory(
    doc_id="paper_001_concept_A",
    document="The Vision Transformer (ViT) model applies the Transformer architecture, " \
             "commonly used in NLP, to image classification tasks by splitting images into patches.",
    metadata={"source": "Attention Is All You Need", "domain": "Vision"}
)

store_memory(
    doc_id="protein_folding_challenge",
    document="A major challenge in bioinformatics is protein structure prediction, which involves " \
             "determining the three-dimensional shape of a protein from its amino acid sequence.",
    metadata={"source": "AlphaFold Paper", "domain": "Biology"}
)

# Now, let's say your agent needs to solve a biology problem and wants to look for cross-domain inspiration.
print("\n🔍 Querying for inspiration on sequence-based problems...")
relevant_memories = retrieve_memories("How can I solve a problem that involves long sequences of data?")

print("\n✅ Found relevant memories:")
for memory in relevant_memories:
    print(f"- {memory}")