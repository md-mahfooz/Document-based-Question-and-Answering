# Document-based-Question-and-Answering

A full-stack, local-first web application that allows users to upload PDF documents and interact with them using advanced Retrieval-Augmented Generation (RAG). 

Unlike standard linear RAG pipelines, this application uses **Agentic Routing** to classify the user's intent and dynamically execute different LangChain workflows.

## Features
* **Dynamic Intent Routing:** Uses a Pydantic-structured LLM parser to determine if the user is asking a question, requesting a summary, or asking for a study guide.
* **Parallel Execution:** Utilizes LangChain's `RunnableParallel` to simultaneously generate notes and quizzes to reduce processing time.
* **Local-First AI:** Powered entirely by open-source models (Llama 3 via Ollama) and local embeddings (Hugging Face / FAISS), ensuring 100% data privacy.
* **Streamlit UI:** A clean, responsive web interface equipped with memory caching to prevent redundant document processing.

## Tech Stack
* **Framework:** LangChain (LCEL)
* **Frontend:** Streamlit
* **LLM:** Llama 3 (via Ollama)
* **Embeddings:** `all-MiniLM-L6-v2` (Hugging Face)
* **Vector Store:** FAISS

## How to Run Locally
1. Ensure you have [Ollama](https://ollama.com/) installed and run `ollama pull llama3`.
2. Clone this repository.
3. Install dependencies: `pip install -r requirements.txt`
4. Launch the app: `streamlit run app.py`
