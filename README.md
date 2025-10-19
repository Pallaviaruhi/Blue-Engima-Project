Here is the full `README.md` content with the "Example Interaction" sections correctly formatted as code blocks for you to copy and paste.

````markdown
# Hybrid AI Travel Assistant 🇻🇳

This project is an advanced Retrieval-Augmented Generation (RAG) system designed to answer travel queries about Vietnam. It leverages a hybrid data retrieval approach, combining the strengths of semantic vector search with a structured knowledge graph to provide contextually rich, accurate, and engaging answers.

This repository contains **two complete implementations** of the chat assistant:
1.  **`hybrid_chat.py` (Manual Build):** A from-scratch implementation using Python's `asyncio` to demonstrate foundational understanding of RAG and asynchronous programming.
2.  **`hybrid_chat_langchain.py` (Framework Build):** A modern, modular implementation using **LangChain Expression Language (LCEL)** to show a scalable, maintainable, and industry-standard approach.

## Key Features

* **Hybrid Data Retrieval:** Uses **Pinecone** for fast semantic search (finding *what's* relevant) and **Neo4j** to enrich that context with factual relationships (finding *how* it's connected).
* **Dual Implementation:** Provides two complete versions of the app, allowing for a direct comparison between a manual build and a modern framework build.
* **Advanced Prompt Engineering:** Both versions use an advanced prompt that gives the AI an "expert travel guide" persona and includes strict instructions to *not* use technical IDs, resulting in clean, natural, and human-readable answers.
* **Performance Optimized (Manual Build):** The `hybrid_chat.py` version features:
    * **Non-Blocking Async Execution:** Uses `asyncio` and `asyncio.to_thread` to run all I/O-bound operations.
    * **Embedding Cache:** Caches OpenAI embedding requests in-memory to reduce API calls.
* **Modular & Extensible (LangChain Build):** The `hybrid_chat_langchain.py` version features:
    * **LCEL Chain:** The entire RAG pipeline is defined in a few readable lines of code.
    * **Standard Components:** Uses official `PineconeVectorStore` and `Neo4jGraph` integrations.
    * **Secure Configuration:** Uses `python-dotenv` to load credentials from a `.env` file, keeping secrets out of the source code.

## System Architecture

The application follows a multi-step RAG pipeline to generate answers:

1.  **Query Embedding:** The user's natural language query is converted into a vector embedding.
2.  **Semantic Search:** The query vector is used to search the **Pinecone** index to find the most semantically similar travel nodes.
3.  **Graph Enrichment:** The IDs of the nodes from Pinecone are used to query the **Neo4j** knowledge graph, finding explicitly connected entities and their relationships.
4.  **Context Compilation:** The results from both Pinecone and Neo4j are compiled into a single, detailed prompt.
5.  **Answer Generation:** The prompt is sent to an OpenAI chat model, which generates a final, human-readable answer based *only* on the provided context.

## Technology Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LangChain](https://img.shields.io/badge/LangChain-18232C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-0B5CD2?style=for-the-badge&logo=pinecone&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-018bff?style=for-the-badge&logo=neo4j&logoColor=white)

## Getting Started

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
    # For Windows
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    Run the following command to install all necessary packages for *both* versions of the script.
    ```bash
    pip install pinecone openai neo4j tqdm langchain langchain-openai langchain-pinecone langchain-neo4j python-dotenv
    ```

### 3. Configuration (Choose one)

You only need to configure the version you want to run.

#### Option 1: For `hybrid_chat.py` (Manual Build)

1.  Rename `config.py.sample` to `config.py`.
2.  Open `config.py` and add your API keys. **Important:** Use the `neo4j+ssc://` scheme for your URI to avoid SSL errors.

    ```python
    # config.py
    OPENAI_API_KEY = "sk-..."
    PINECONE_API_KEY = "..."
    PINECONE_INDEX_NAME = "vietnam-travel"
    PINECONE_VECTOR_DIM = 1536

    NEO4J_URI = "neo4j+ssc://YOUR-AURA-ID.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "..."
    ```

#### Option 2: For `hybrid_chat_langchain.py` (Recommended)

1.  Create a new file in the root folder named `.env`
2.  Add your credentials to this file. LangChain will automatically load them.

    ```
    # .env file
    OPENAI_API_KEY="sk-..."
    PINECONE_API_KEY="..."
    PINECONE_INDEX_NAME="vietnam-travel"

    NEO4J_URI="neo4j+s://YOUR-AURA-ID.databases.neo4j.io"
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="YOUR-AURA-PASSWORD"
    ```

### 4. Data Loading

This step is **required once** and works for both versions. Make sure your `config.py` file is set up before running.

1.  **Load data into Neo4j:**
    ```bash
    python load_to_neo4j.py
    ```
2.  **Generate embeddings and upload to Pinecone:**
    ```bash
    python pinecone_upload.py
    ```

### 5. Running the Assistant

You can now run either version of the chat assistant.

#### Version 1: Manual Build
```bash
python hybrid_chat.py
````

**Example Interaction:**

```bash
Hybrid travel assistant. Type 'exit' to quit.

Enter your travel question: create a romantic 4 day itinerary for Vietnam
DEBUG: Cache miss. Calling API.
DEBUG: Pinecone found 5 matches.
DEBUG: Neo4j found 23 graph facts.

=== Assistant Answer ===

Absolutely! Here’s a romantic 4-day itinerary for you and your partner...

=== End ===
```

#### Version 2: LangChain Build

```bash
python hybrid_chat_langchain.py
```

**Example Interaction:**

```bash
--- LangChain Hybrid RAG Assistant ---
Type 'exit' to quit.

Enter your travel question: create a romantic 4 day itinerary for Vietnam

=== Assistant Answer ===

Absolutely! Here’s a romantic 4-day itinerary for you to enjoy the beauty and charm of Vietnam.

**Day 1: Arrival in Da Lat**
Start your romantic journey in Da Lat...

=== End ===
```

```
```
