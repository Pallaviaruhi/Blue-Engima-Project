# AI-Hybrid Chat Evaluation: Project Write-up

This document explains the steps I took to get the AI-Hybrid Chat project working, debug all the issues, and add bonus features to improve its performance and the quality of its answers.

## Part 1: System Setup and Data Ingestion

This foundational phase involved preparing the environment and populating both the graph and vector databases.

* **Environment Configuration:** Created accounts for OpenAI, Pinecone, and Neo4j. I put all the API keys and login details into the `config.py` file so the other scripts could easily connect to these services.
* **Graph Database Setup (Neo4j):** Before the `load_to_neo4j.py` script could run, I had to fix a major connection error.
    * **Neo4j Connection Error:** The script failed with an `ssl.SSLCertVerificationError`. I diagnosed this as an SSL certificate validation issue when connecting to the Neo4j Aura cloud database from my local machine.
    * **Fix:** I updated the `NEO4J_URI` in the configuration to use the `neo4j+ssc://` scheme. This tells the driver to establish a secure connection but to skip the certificate chain verification that was failing.
* After fixing the connection, the script ran successfully, and I used the Neo4j web dashboard to confirm that all 360 nodes and 370 relationships were loaded correctly.
* **Vector Database Setup (Pinecone):** This part had two problems:
    1.  **Dependency Fix:** The project wouldn't run because it used an old package called `pinecone-client`. I fixed this by running `pip uninstall pinecone-client` and then `pip install pinecone` to get the correct, modern version.
    2.  **API Money Error:** The script kept failing with an "insufficient\_quota" error. I realized this wasn't a code bug, but that my OpenAI account needed credits to pay for making the embeddings. After I added a payment method, the script worked perfectly and uploaded all the data.

---

## Part 2: Core Application Debugging (`hybrid_chat.py`)

With the data loaded, I focused on making the core `hybrid_chat.py` script functional. The script crashed at first because it was written for an old version of the Pinecone SDK.

* **Fixing Pinecone SDK v3+ Errors:** I had to update the code in two places:
    1.  **Checking the Index:** The old way of checking if an index exists (`.names()`) doesn't work anymore. I updated it to the new way, which involves getting a list of index details.
    2.  **Reading Query Results:** The old code expected a Python dictionary (`res['matches']`) back from Pinecone. The new SDK returns an object, so I fixed the code to use dot notation (`res.matches`) to read the results.

---

## Part 3: Bonus Improvements (For a Better Grade)

After the chat was *functional*, I added three major improvements to make it *excellent*.

### 1. Superior Prompt Engineering (The "Chain of Thought" Prompt)

This was the most important change. The initial prompt was basic. To get a better answer, I implemented an advanced "Chain of Thought" technique:

* **System Prompt:** I gave the AI a new persona: a "world-class travel expert" whose job is to be friendly and enthusiastic. I strictly *forbade* it from using any technical IDs or citations in its final answer.
* **User Prompt:** I created a special template. I instructed the AI to first write an **"[Internal Scratchpad]"** (where it does all its planning) and then a **"[Final Answer]"** (which is clean and conversational).
* **Fixing the Token Limit:** This new "Chain of Thought" prompt required the AI to write *a lot* more text (both its thoughts and the final answer). This caused the response to be cut off. I diagnosed this as the `max_tokens=600` limit being too low. **I fixed this by increasing the limit to `1500` in the `call_chat` function**, giving the AI enough space to complete its complex response.
* **Parsing Code:** I added a `parse_response` function to automatically split the AI's full response and *only* show the user the clean "[Final Answer]" part.

### 2. Request Caching for Embeddings

Every time I asked the same question, the app would call the OpenAI API to get an embedding, which costs money and time. I fixed this by adding a simple dictionary called `embedding_cache`. Now, before making an API call, the code checks if the embedding is already in the cache. If it is, it uses the cached version instantly. This makes repeated queries much faster and cheaper.

### 3. Asynchronous Execution and Code Readability

* **Asynchronous Design:** I used Python's `asyncio` library to build the main chat loop. This makes the application more robust. Instead of freezing the whole program while waiting for the database, it uses `asyncio.to_thread`. This means it can handle multiple users at once without getting stuck, which is a much better design for a real application.
* **Adding Code Comments:** To make the project easy to understand and grade, I went through the entire `hybrid_chat.py` file and added detailed comments.
    * **Docstrings:** I added `"""..."""` blocks to every function to explain what it does, what its parameters are, and what it returns.
    * **Inline Comments:** I added `#` comments to explain *why* certain lines of code are there (like explaining the `max_tokens` fix or why we use `asyncio.to_thread`). This makes the code readable and fulfills the "Design & Readability" part of the rubric.

---

## Part 4: Bonus Innovation (LangChain Implementation)

After building the RAG system manually in `hybrid_chat.py`, To provide a more comprehensive solution and demonstrate how this same system could be built using an industry-standard framework: **LangChain**.

I created a new file, **`hybrid_chat_langchain.py`**, which builds the *exact same* RAG pipeline but with far less code. This new script demonstrates a more modular, readable, and scalable approach.

### Key Changes & Improvements:

* **Modularity:** All the manual "plumbing" code was replaced with standard, swappable LangChain components.
    * `ChatOpenAI` and `OpenAIEmbeddings` for the models.
    * `PineconeVectorStore.from_existing_index` to create a retriever.
    * The official `Neo4jGraph` component to manage the graph connection.

* **Superior Readability (LCEL):** The biggest improvement. All the complex `asyncio` logic and manual function-calling from the first script was completely replaced by a single, declarative **LCEL (LangChain Expression Language) chain**. The flow of data is now perfectly clear and easy to read:

    ```python
    chain = retriever_chain | context_chain | prompt_template | llm | StrOutputParser()
    ```

* **Flexibility (RunnableLambda):** I was still able to use my own custom Python functions. I wrapped my `get_graph_context` and `format_vector_context` functions inside a `RunnableLambda`. This shows how LangChain can easily mix and match pre-built components with custom code.

* **Better Configuration:** This version uses `python-dotenv` to load all API keys and credentials from a `.env` file. This is a much cleaner and more secure practice than using a `config.py` file, as the credentials are not part of the source code.    