# Hybrid AI Travel Assistant 🇻🇳

This project is an advanced Retrieval-Augmented Generation (RAG) system designed to answer travel queries about Vietnam. It leverages a hybrid data retrieval approach, combining the strengths of semantic vector search with a structured knowledge graph to provide contextually rich, accurate, and engaging answers.

##  Key Features

* **Hybrid Data Retrieval:** Uses **Pinecone** for fast semantic search to find relevant locations and activities, and **Neo4j** to enrich this context with relational data (e.g., nearby hotels, other attractions in the same city).
* **Advanced Prompt Engineering:** Implements a detailed system prompt that gives the AI a "world-class travel expert" persona, ensuring high-quality, well-structured, and helpful responses.
* **Performance Optimized:**
    * **Asynchronous Execution:** Uses Python's `asyncio` to run Pinecone and Neo4j queries in parallel, significantly reducing response latency.
    * **Embedding Cache:** Caches OpenAI embedding requests in memory to eliminate redundant API calls, speeding up repeated queries and reducing costs.
* **Modern Tech Stack:** Built with the latest versions of the Pinecone and OpenAI Python SDKs.

##  System Architecture

The application follows a multi-step RAG pipeline to generate answers:

1.  **Query Embedding:** The user's natural language query (e.g., "create a romantic 4 day itinerary for Vietnam") is converted into a vector embedding using OpenAI's `text-embedding-3-small` model.
2.  **Semantic Search:** The query vector is used to search the **Pinecone** index, retrieving the `top_k` most semantically similar travel nodes (attractions, activities, etc.).
3.  **Graph Enrichment:** The IDs of the nodes retrieved from Pinecone are used to query the **Neo4j** knowledge graph, finding explicitly connected entities and their relationships.
4.  **Context Compilation:** The results from both Pinecone (semantic context) and Neo4j (relational context) are compiled into a single, detailed prompt.
5.  **Answer Generation:** The compiled prompt is sent to an OpenAI chat model (`gpt-4o-mini`), which generates a final, human-readable answer based *only* on the provided context.



##  Technology Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-0B5CD2?style=for-the-badge&logo=pinecone&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-018bff?style=for-the-badge&logo=neo4j&logoColor=white)

##  Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

* Python 3.8+
* Accounts for:
    * [OpenAI](https://platform.openai.com/)
    * [Pinecone](https://www.pinecone.io/)
    * [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-database/) (or a local instance)

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    *(First, ensure you have a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment after installing the packages (`pinecone`, `openai`, `neo4j`, `tqdm`).)*
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration

1.  Rename `config.py.sample` to `config.py`.
2.  Open `config.py` and add your API keys and credentials for OpenAI, Pinecone, and Neo4j.

    ```python
    # config.py
    OPENAI_API_KEY = "sk-..."
    PINECONE_API_KEY = "..."
    PINECONE_INDEX_NAME = "vietnam-travel"
    PINECONE_VECTOR_DIM = 1536 # Based on text-embedding-3-small

    NEO4J_URI = "neo4j+s://..."
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "..."
    ```

### 4. Data Loading

You must run the data loading scripts in the correct order.

1.  **Load data into Neo4j:**
    ```bash
    python load_to_neo4j.py
    ```
2.  **Generate embeddings and upload to Pinecone:**
    ```bash
    python pinecone_upload.py
    ```

### 5. Running the Assistant

Once the setup is complete, you can start the interactive chat assistant.

```bash
python hybrid_chat.py
```

You will be prompted to enter your travel questions in the terminal.

**Example Interaction:**
```
Hybrid travel assistant. Type 'exit' to quit.

Enter your travel question: create a romantic 4 day itinerary for Vietnam
DEBUG: Cache miss. Calling API.
DEBUG: Pinecone top 5 results:
5
DEBUG: Graph facts:
23

=== Assistant Answer ===

Absolutely! Here’s a romantic 4-day itinerary for you and your partner...

=== End ===
```
