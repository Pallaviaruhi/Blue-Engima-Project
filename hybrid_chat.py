# -----------------------------
# Imports
# -----------------------------
import json
from typing import List
import asyncio  # Used for running database queries concurrently (asynchronously)
import config   # Our local file for storing API keys and configs

# --- Third-party libraries ---
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase

# -----------------------------
# Global Cache & Config
# -----------------------------

# In-memory cache to store generated embeddings.
# This avoids costly and slow API calls for repeated queries.
embedding_cache = {}

# --- Model Configuration ---
EMBED_MODEL = "text-embedding-3-small"  # Model for creating vector embeddings
CHAT_MODEL = "gpt-4o-mini"              # Model for generating the final chat answer
TOP_K = 5                               # Number of semantic results to fetch from Pinecone
INDEX_NAME = config.PINECONE_INDEX_NAME # Name of our Pinecone index

# -----------------------------
# Initialize API Clients
# -----------------------------

# Initialize the OpenAI client using the API key from our config file
print("Initializing API clients...")
client = OpenAI(api_key=config.OPENAI_API_KEY)

# Initialize the Pinecone client
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# --- Connect to Pinecone Index ---
# Check if the specified index already exists in our Pinecone project
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    # If it doesn't exist, create it.
    # This is crucial for the first run.
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM, # 1536 for text-embedding-3-small
        metric="cosine", # Cosine similarity is good for text embeddings
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Get a handle to the specific index we'll be querying
index = pc.Index(INDEX_NAME)

# --- Connect to Neo4j Database ---
# Initialize the Neo4j graph database driver
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)
print("Clients initialized and databases connected.")

# -----------------------------
# Helper Functions
# -----------------------------

def embed_text(text: str) -> List[float]:
    """
    Gets a vector embedding for a text string, using a cache.
    
    :param text: The user's query string to embed.
    :return: A list of floats representing the vector embedding.
    """
    # 1. Check if we've already generated and cached this embedding
    if text in embedding_cache:
        print("DEBUG: Cache hit!")
        return embedding_cache[text]
    
    # 2. If not in cache, call the OpenAI API
    print("DEBUG: Cache miss. Calling API.")
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    embedding = resp.data[0].embedding
    
    # 3. Store the new embedding in our cache before returning it
    embedding_cache[text] = embedding
    return embedding

def pinecone_query(query_text: str, top_k=TOP_K):
    """
    Queries the Pinecone index for semantically similar vectors.
    
    :param query_text: The user's natural language query.
    :param top_k: The number of results to return.
    :return: A list of Pinecone match objects.
    """
    # Get the embedding for the user's query (uses our cache)
    vec = embed_text(query_text)
    
    # Perform the vector search in Pinecone
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,  # We need metadata (like 'name')
        include_values=False    # We don't need the raw vector values
    )
    print(f"DEBUG: Pinecone found {len(res.matches)} matches.")
    # The new SDK returns results in a 'matches' attribute
    return res.matches

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """
    Fetches neighboring nodes and relationships from Neo4j for a given list of node IDs.
    
    :param node_ids: A list of node IDs (e.g., ['attraction_101', 'hotel_50'])
    :param neighborhood_depth: How many "hops" to explore (default is 1)
    :return: A list of "facts" (dictionaries) describing the graph relationships.
    """
    facts = []
    # Use a 'with' block to ensure the Neo4j session is properly closed
    with driver.session() as session:
        # Loop through each node ID found by Pinecone
        for nid in node_ids:
            # This Cypher query finds the node 'n' by its ID,
            # then finds any node 'm' connected to it by any relationship 'r'.
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10" # Limit to 10 neighbors per node to avoid too much context
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                # Format the graph data into a simple dictionary ("fact")
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    # Truncate description to save tokens
                    "target_desc": (r["description"] or "")[:400], 
                    "labels": r["labels"]
                })
    print(f"DEBUG: Neo4j found {len(facts)} graph facts.")
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """
    Builds the advanced 'Chain of Thought' prompt for the LLM.
    This prompt instructs the AI to "think" in a scratchpad first,
    then write a clean, user-facing answer.
    
    :param user_query: The original query from the user.
    :param pinecone_matches: The list of match objects from Pinecone.
    :param graph_facts: The list of fact dictionaries from Neo4j.
    :return: An array of message objects for the OpenAI API.
    """
    
    # 1. System Prompt: Defines the AI's persona, rules, and goals.
    system = (
        "You are a world-class travel expert specializing in creating personalized and engaging itineraries for Vietnam. "
        "Your tone is enthusiastic, helpful, and natural. "
        "You will be given context from a vector search ('semantic matches') and a knowledge graph ('graph facts'). "
        "Your primary goal is to synthesize this information to create a logical, day-by-day itinerary. "
        "IMPORTANT: Your final response to the user must be clean and conversational. "
        "**Do not include any node IDs, scores, metadata, or citations (e.g., '(id: ...)') in your final answer.** "
        "Just use the names of the places and weave them into a beautiful itinerary."
    )

    # 2. Format Pinecone Context
    vec_context = []
    for m in pinecone_matches:
        meta = m.metadata        
        score = m.score            
        snippet = f"- id: {m.id}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    # 3. Format Neo4j Context
    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    # 4. User Prompt: Contains the context and the "Chain of Thought" instructions.
    user_content = (
        f"User query: {user_query}\n\n"
        "--- Context from Vector Database (Semantic Matches) ---\n"
        "\n".join(vec_context[:10]) + "\n\n"
        "--- Context from Knowledge Graph (Graph Facts) ---\n"
        "\n".join(graph_context[:20]) + "\n\n"
        "--- Your Task ---\n"
        "Please follow these two steps:\n"
        "1. **Internal Scratchpad:** First, think step-by-step. Analyze the user's query and the provided context. Identify the key locations, note their relationships from the graph, and plan out a logical itinerary. Write this reasoning down for your own use.\n"
        "2. **Final Answer:** After your scratchpad, write the final, clean, and engaging answer for the user. This answer must be conversational and **must not contain any node IDs, citations, or references to the 'context' or 'scratchpad'.**\n\n"
        "Begin your response *only* with '[Scratchpad]' and then '[Final Answer]'.\n\n"
        "[Scratchpad]\n"
        # The AI will start its response here, first by filling out its thought process.
    )

    # 5. Combine prompts into the final message array for the API
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]
    return prompt

def call_chat(prompt_messages):
    """
    Calls the OpenAI ChatCompletion API with the constructed prompt.
    
    :param prompt_messages: The list of message objects from build_prompt.
    :return: The raw string response from the LLM.
    """
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=1500,
        temperature=0.2 # Low temperature for more predictable, factual answers
    )
    return resp.choices[0].message.content

def parse_response(full_response: str) -> str:
    """
    Extracts just the 'Final Answer' part from the LLM's full output.
    
    :param full_response: The raw string from the LLM, containing both scratchpad and answer.
    :return: The clean, user-facing answer string.
    """
    # Check for our [Final Answer] tag
    if "[Final Answer]" in full_response:
        # Split the string on the tag and take the *last* part
        return full_response.split("[Final Answer]")[-1].strip()
    else:
        # Fallback in case the model forgets to use the tag
        return full_response.strip()

# -----------------------------
# Asynchronous Main Chat Loop
# -----------------------------

async def process_query(query: str):
    """
    Asynchronously processes a single user query from start to finish.
    This runs the I/O-bound database calls in parallel.
    
    :param query: The user's input string.
    """
    
    # 1. Run I/O-bound tasks (database queries) in separate threads
    #    so they don't block the main async loop.
    #    asyncio.to_thread is the modern way to do this.
    
    # Create tasks for Pinecone and Neo4j queries
    pinecone_task = asyncio.to_thread(pinecone_query, query, top_k=TOP_K)
    
    # Wait for the Pinecone query to finish first, since Neo4j depends on it
    matches = await pinecone_task
    
    # Once we have matches, we can start the Neo4j query
    match_ids = [m.id for m in matches]
    graph_facts_task = asyncio.to_thread(fetch_graph_context, match_ids)
    
    # Wait for the Neo4j query to finish
    graph_facts = await graph_facts_task

    # 2. Build the prompt (CPU-bound, no 'await' needed)
    prompt = build_prompt(query, matches, graph_facts)
    
    # 3. Call the LLM (I/O-bound, so run in a thread)
    full_answer_string = await asyncio.to_thread(call_chat, prompt)
    
    # 4. Parse the final clean answer (CPU-bound)
    clean_answer = parse_response(full_answer_string)
    
    # 5. Print the result to the user
    print("\n=== Assistant Answer ===\n")
    print(clean_answer)
    print("\n=== End ===\n")

async def interactive_chat():
    """
    Main async loop to handle user input.
    """
    print("Hybrid travel assistant. Type 'exit' to quit.")
    while True:
        # Run the blocking 'input()' function in a separate thread
        # to avoid freezing the async event loop.
        query = await asyncio.to_thread(input, "\nEnter your travel question: ")
        query = query.strip()
        
        if not query or query.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break
        
        # Wait for the entire query-to-answer pipeline to complete
        await process_query(query)

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    # This is the main entry point of the script.
    # asyncio.run() starts the asynchronous event loop and runs
    # our 'interactive_chat' function.
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\nChat manually interrupted. Exiting.")
