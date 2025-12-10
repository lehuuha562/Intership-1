# 🔍 Vector Database Search System

### Internship 1 Project | HCMC University of Technology (HCMUT)

**Author:** Le Huu Ha  
**Student ID:** 2470489  
**Supervisor:** Prof. Thoai Nam  
**Version:** OpenWork Vol. 3

-----

## 📖 Overview

[cite_start]This project is a **local, privacy-focused multimodal search engine** designed to overcome the limitations of traditional keyword search (BM25)[cite: 5, 14, 15]. [cite_start]It leverages **Vector Embeddings** for semantic understanding and **Retrieval-Augmented Generation (RAG)** to provide natural language answers[cite: 6, 7].

[cite_start]The system is built on a **Microservices Architecture** fully containerized with Docker, ensuring easy deployment and reproducibility[cite: 6, 28].

## 🏗️ System Architecture & Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit (Python) | [cite_start]Reactive web interface for search and visualization[cite: 7, 53]. |
| **Vector Database** | Milvus 2.3 (Standalone) | [cite_start]Stores 384-dim vectors; uses **HNSW** indexing for high-speed retrieval (\~12ms)[cite: 7, 56, 150]. |
| **Inference Engine** | Ollama | [cite_start]Hosts the **Phi-3 Mini (3.8B)** Large Language Model for RAG[cite: 7, 61, 109]. |
| **Embedding Model** | Sentence-BERT | [cite_start]`all-MiniLM-L6-v2` model for converting text to vectors[cite: 66]. |
| **Storage & Meta** | MinIO & Etcd | [cite_start]Object storage for vectors and metadata coordination[cite: 58, 59]. |
| **Visualization** | t-SNE | [cite_start]Dimensionality reduction for the interactive 2D "Galaxy View"[cite: 8, 131]. |

## 📋 Prerequisites

You do NOT need Python or Milvus installed on your host machine.

  * [cite_start]**[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Required)[cite: 26].
  * *Recommended:* 8GB+ RAM (for running the LLM and Database simultaneously).

## 🚀 Quick Start

1.  **Open Docker Desktop**: Ensure the Docker engine is running in the background.
2.  [cite_start]**Add Data**: Place your `books.csv` inside the `data/` folder[cite: 67].
3.  **Start the System**:
    Double-click the `run.bat` file in the project folder (or run it via terminal):
    ```bash
    run.bat
    ```
    *Note: The first run may take a few minutes to pull the Ollama images and the Phi-3 model.*
4.  **Access the Application**:
      * **Frontend UI:** [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)
      * **Milvus Admin:** [http://localhost:19530](https://www.google.com/search?q=http://localhost:19530) (if Attu is included)

## 🛠️ Key Features

### 1\. Semantic Search (RAG Pipeline)

[cite_start]Unlike standard keyword search, you can query by **meaning**[cite: 23].

  * **Query:** *"Stories about giant lizards"*
  * [cite_start]**Retrieves:** *"The Book of Dragons"* (Matches conceptually even without keyword overlap)[cite: 155, 156].
  * [cite_start]**RAG Generation:** The Phi-3 model reads the retrieved book summaries and answers your question naturally[cite: 7, 104].

### 2\. Galaxy View (Data Visualization)

Click the **"Visualize"** tab to see your entire dataset.

  * [cite_start]Uses **t-SNE** to project high-dimensional vectors (384-d) into a 2D interactive scatter plot[cite: 8, 131].
  * [cite_start]Colors indicate clusters of semantically similar documents[cite: 25].

### 3\. High-Performance Indexing

  * [cite_start]**Index Type:** HNSW (Hierarchical Navigable Small World)[cite: 8].
  * [cite_start]**Metric:** Cosine Similarity[cite: 70].
  * [cite_start]**Performance:** Reduced P99 search latency from 150ms (Legacy) to **12ms**[cite: 9, 150].

## 📂 Project Structure
'''
[cite_start]├── app.py                 # Streamlit Frontend entry point [cite: 53]
[cite_start]├── embedder.py            # Sentence-BERT logic for text ingestion [cite: 66]
[cite_start]├── rag_engine.py          # Logic for communicating with Ollama (Phi-3) [cite: 32]
├── run.bat                # Script to start the system
├── close.bat              # Script to stop the system
[cite_start]├── docker-compose.yml     # Orchestration of Milvus, Etcd, MinIO, and App [cite: 47]
├── Dockerfile             # Build instructions for the Python app
├── data/
│   └── books.csv          # Source dataset
└── utils.py               # Database connection helpers
```

## 🛑 Stopping the System

To shut down all containers and clean up resources, simply run:

```bash
close.bat
```

## ❓ Troubleshooting

**"Conflict: Container name already in use"**
If `run.bat` fails because containers are already running, try running `close.bat` first to clean up the previous session.

**"Ollama Connection Refused"**
Ensure the `ollama-container` is healthy. [cite_start]It runs on internal port `11434`[cite: 38]. If using RAG, wait 30 seconds after startup for the model to load into memory.
