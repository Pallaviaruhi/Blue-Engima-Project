# hybrid_chat.py
import json
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
import asyncio


# Added a dictionary to store embeddings as Cache
embedding_cache = {}


# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1") # Note: Changed region to match pinecone region
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# -----------------------------
# Helper functions
# -----------------------------
# This is NEW function with caching
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string, using a cache."""
    if text in embedding_cache:
        print("DEBUG: Cache hit!")
        return embedding_cache[text]

    print("DEBUG: Cache miss. Calling API.")
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    embedding = resp.data[0].embedding
    embedding_cache[text] = embedding # Store embedding in the cache
    return embedding

def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print("DEBUG: Pinecone top 5 results:")
    print(len(res.matches))
    return res.matches

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })
    print("DEBUG: Graph facts:")
    print(len(facts))
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
    "You are a world-class travel expert specializing in creating personalized and engaging itineraries for Vietnam. "
    "Your tone is enthusiastic and helpful. "
    "Use the provided 'semantic matches' as the primary inspiration for locations and the 'graph facts' to find connections and nearby activities. "
    "Your primary goal is to create a logical, day-by-day itinerary based ONLY on the provided context. "
    "For each suggested activity, ALWAYS cite its node ID like this: (id: attraction_123). "
    "Do not invent places or activities not mentioned in the context. "
    "Conclude your answer with a friendly closing remark."
)

    vec_context = []
    for m in pinecone_matches:
        meta = m.metadata           # Used dot notation
        score = m.score             
        snippet = f"- id: {m.id}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score}" # Use m.id
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         "Top semantic matches (from vector DB):\n" + "\n".join(vec_context[:10]) + "\n\n"
         "Graph facts (neighboring relations):\n" + "\n".join(graph_context[:20]) + "\n\n"
         "Based on the above, answer the user's question. If helpful, suggest 2–3 concrete itinerary steps or tips and mention node ids for references."}
    ]
    return prompt

def call_chat(prompt_messages):
    """Call OpenAI ChatCompletion."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=600,
        temperature=0.2
    )
    return resp.choices[0].message.content


# -----------------------------
# Interactive chat (Asynchronous Version)
# -----------------------------
async def process_query(query: str):
    """Asynchronously processes a single user query."""

    # Step 1: Get Pinecone matches (this is I/O bound, run in a thread)
    matches = await asyncio.to_thread(pinecone_query, query, top_k=TOP_K)

    # Step 2: Get graph context (this is I/O bound, run in a thread)
    match_ids = [m.id for m in matches]
    graph_facts = await asyncio.to_thread(fetch_graph_context, match_ids)

    # Step 3: Build the prompt (this is CPU bound, can run normally)
    prompt = build_prompt(query, matches, graph_facts)

    # Step 4: Call the Chat model (this is I/O bound, run in a thread)
    answer = await asyncio.to_thread(call_chat, prompt)

    print("\n=== Assistant Answer ===\n")
    print(answer)
    print("\n=== End ===\n")


async def interactive_chat():
    """Main async chat loop."""
    print("Hybrid travel assistant. Type 'exit' to quit.")
    while True:
        query = await asyncio.to_thread(input, "\nEnter your travel question: ")
        query = query.strip()

        if not query or query.lower() in ("exit", "quit"):
            break

        await process_query(query)

if __name__ == "__main__":
    asyncio.run(interactive_chat())
