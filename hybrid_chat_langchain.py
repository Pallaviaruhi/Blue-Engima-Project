from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_neo4j import Neo4jGraph
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
import os 
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
graph = Neo4jGraph()

vector_store = PineconeVectorStore.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX_NAME"),  # <-- Reads from environment
    embedding=embeddings,
    text_key="name"
    

)

vector_retriever = vector_store.as_retriever(k=5)

# This is our custom function to query Neo4j
def get_graph_context(docs):
    """
    Takes the retrieved documents from Pinecone, extracts their IDs,
    and queries Neo4j for related information.
    """
    # Extract the 'id' from the metadata of each retrieved document
    ids = [doc.metadata['id'] for doc in docs]
    
    # Run a Cypher query to find neighboring nodes
    cypher_query = """
    UNWIND $ids AS entityId
    MATCH (n) WHERE n.id = entityId
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN 
        n.name AS source_name, 
        type(r) AS rel, 
        m.id AS target_id, 
        m.name AS target_name, 
        m.description AS target_desc
    LIMIT 20
    """
    
    # Run the query
    results = graph.query(cypher_query, params={'ids': ids})
    
    # Format the results into a string for the prompt
    graph_context = []
    for f in results:
        snippet = f"- ({f['source_name']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        graph_context.append(snippet)
        
    return "\n".join(graph_context)

def format_vector_context(docs):
    """Formats the vector search results into a string."""
    vec_context = []
    for m in docs:
        meta = m.metadata
        snippet = f"- id: {meta['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)
    return "\n".join(vec_context)

# --- 3. Define the Prompt Template ---

system_prompt = (
    "You are an expert Vietnam travel guide. Your goal is to create a helpful, human-readable itinerary. "
    "Use the provided context to answer the user's question."
    
    "\n\n*Instructions:*\n"
    "- Do not mention 'context', 'semantic matches', or 'graph facts' in your answer. "
    "- *Crucially:* When you mention a place (like an attraction, hotel, or city), "
    "  use its full 'name' (e.g., 'Da Lat Flower Gardens'). "
    "- *Do not* use the place's 'id' (e.g., 'Attraction 250'). "
    "- Be friendly, clear, and structure the answer logically."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", 
     "Here is all the context I've gathered to help you:\n\n"
     
     "Top semantic matches (from vector DB):\n"
     "{vector_context}\n\n"
     
     "Graph facts (neighboring relations):\n"
     "{graph_context}\n\n"
     
     "User query: {user_query}\n\n"
     
     "Based on the context and the user query, please provide your answer. "
     "Remember to follow all instructions from the system role (especially using names, not IDs)."
    )
])


# --- 4. Build the RAG Chain using LCEL ---

from operator import itemgetter  # <-- Make sure to add this import at the top of your file

# Define the first step: run the retriever and pass along the query
# The input is the user_query (string)
retriever_chain = RunnableParallel(
    docs=vector_retriever,
    user_query=RunnablePassthrough() 
    # Output of this step: {'docs': [docs], 'user_query': 'my query'}
)

# Define the second step: format the context
# The input is the dict from the step above
context_chain = RunnableParallel(
    vector_context=itemgetter("docs") | RunnableLambda(format_vector_context),
    graph_context=itemgetter("docs") | RunnableLambda(get_graph_context),
    user_query=itemgetter("user_query")
)

# This is the final chain!
# 1. Run retriever_chain
# 2. Pass its output to context_chain
# 3. Pass its output to the prompt_template
# 4. Pass the prompt to the llm
# 5. Parse the output to a string
chain = retriever_chain | context_chain | prompt_template | llm | StrOutputParser()

# --- 5. Create the Interactive CLI ---

if __name__ == "__main__":
    print("--- LangChain Hybrid RAG Assistant ---")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nEnter your travel question: ")
            if query.lower() == 'exit':
                break
            
            # Invoke the chain!
            result = chain.invoke(query)
            
            print("\n=== Assistant Answer ===")
            print(result)
            print("=== End ===")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")