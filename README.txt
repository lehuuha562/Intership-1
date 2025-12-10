# 🔍 Vector Database Search

This project is a containerized search engine capable of indexing and searching through **Books (CSV)**, **Images**, **Videos**, and **Audio** files using vector similarity. It uses **Milvus** for the vector database and **Streamlit** for the web interface, all wrapped in a single Docker environment.

## 📋 Prerequisites

You only need **one** thing installed on your computer:

  * **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Windows, Mac, or Linux)

*No Python, PyTorch, or Database installation is required on your host machine.*

## 🚀 Quick Start (How to Run)

1.  **Download & Extract** the project folder.
2.  **Add your Data**:
      * Place your `books.csv` or any media files (images, videos) inside the `data/` folder.
3.  **Open Terminal** (Command Prompt, PowerShell, or Mac Terminal) and navigate to the project folder:
    ```bash
    cd path/to/project
    ```
4.  **Start the System**:
    Run this single command to build the app and start the database:
    ```bash
    docker-compose up --build
    ```
5.  **Access the App**:
    Wait for the logs to stop scrolling, then open your browser to:
    👉 **http://localhost:8501**

## 📂 Project Structure

  * **`app.py`**: The frontend web interface (Streamlit).
  * **`data/`**: Place your input files here. The app automatically detects if you have a CSV (Book Mode) or a folder of media (Multimodal Mode).
  * **`docker-compose.yml`**: Orchestrates the Milvus database and the Python application containers.
  * **`utils.py`**: Handles database connections and automatic data ingestion.

## 🛠️ How to Use

### 1\. Book Search Mode

  * Ensure a file named `books.csv` is in the `data/` folder.
  * The app will automatically ingest the titles and genres.
  * Type queries like *"fantasy novels about dragons"* in the search bar.

### 2\. Multimodal Search Mode (Images/Video/Audio)

  * Place a folder of images or videos inside `data/`.
  * The system uses `embedder.py` to scan the folder and generate vector embeddings for visual and audio content.
  * You can search for visual concepts (e.g., *"sunset over the ocean"*) to find matching images or video frames.

## 🛑 Stopping the App

To stop the server and shut down the database safely, press `Ctrl+C` in your terminal, or run:

```bash
docker-compose down
```

## ❓ Troubleshooting

**"Conflict: Container name already in use"**
If you see this error, it means a previous session didn't close properly. Force delete the old containers:

```bash
docker rm -f milvus-etcd milvus-minio milvus-standalone search-app
```

**"Failed to connect to Milvus"**

Ensure you are running the app via `docker-compose up`. Do not try to run `streamlit run app.py` directly on your local machine, as it won't be able to see the database inside the Docker network.
