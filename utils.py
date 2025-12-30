# utils.py
import json
import os
import time
from pymilvus import connections, utility, Collection
from rich.console import Console
from typing import Any

console = Console()

VECTOR_DIM = 512  
COLLECTION_NAME = "demo_embeddings"

def save_json(data: Any, path: str):
    """Save data to JSON file (Restored)"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        console.print(f"[red]Failed to save JSON to {path}: {e}[/red]")
        raise

def load_json(path: str) -> Any:
    """Load data from JSON file (Restored)"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
# ------------------------------------

def connect_milvus(host: str = None, port: str = "19530"):
    """Connect to Milvus server with retry logic"""
    if host is None:
        host = os.getenv("MILVUS_HOST", "127.0.0.1")
    
    # Disconnect stale connections
    try:
        if connections.has_connection("default"):
            connections.disconnect("default")
    except: pass

    max_retries = 20
    for i in range(max_retries):
        try:
            console.print(f"[cyan]Connecting to Milvus at {host}:{port} ({i+1})...[/cyan]")
            connections.connect(alias="default", host=host, port=port)
            console.print(f"[green]Success![/green]")
            return
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                raise

def prepare_collection(data_path: str):
    # 1. Connect
    connect_milvus()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found at {data_path}")

    # Local imports to avoid circular dependency
    from embedder import embed_books_csv, generate_embeddings
    from searcher import setup_collection

    # 2. Setup Collection (Defaults to HNSW inside searcher.py)
    collection = setup_collection()
    
    # 3. Check if empty, then ingest
    if collection.num_entities == 0:
        console.print(f"[yellow]Collection empty. Ingesting {data_path}...[/yellow]")
        
        is_csv = data_path.lower().endswith('.csv')
        
        if is_csv:
            embed_books_csv(data_path)
        elif os.path.isdir(data_path):
            output_json = "data_embeddings.json"
            generate_embeddings(data_path, output_file=output_json)
            # Load and insert...
            embeddings = load_json(output_json)
            if embeddings:
                ids = [item["id"] for item in embeddings]
                vectors = [item["vector"] for item in embeddings]
                metas = [f"{item['type']} | {item['content']}" for item in embeddings]
                collection.insert([ids, vectors, metas])
    
    collection.flush()
    collection.load()
    return collection