# searcher.py
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType
from rich.console import Console

console = Console()

def setup_collection(collection_name="demo_embeddings", dim=384, index_type="HNSW"):
    """
    Creates the Milvus collection and index.
    """
    if not connections.has_connection("default"):
        raise ConnectionError("No connection to Milvus.")

    if utility.has_collection(collection_name):
        return Collection(collection_name)

    # 1. Define Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="meta", dtype=DataType.VARCHAR, max_length=65535),
    ]
    
    schema = CollectionSchema(fields, "Multimodal Search Collection")
    collection = Collection(collection_name, schema)
    
    # 2. Define Index (Switching to COSINE to match your existing DB)
    index_params = {
        "metric_type": "COSINE",  # <--- CHANGED THIS from L2
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }

    console.print(f"[cyan]Building Index: HNSW (COSINE)...[/cyan]")
    collection.create_index(field_name="vector", index_params=index_params)
    console.print("[green]Index Built![/green]")
    
    return collection

def perform_search(collection, query_vectors, limit=5, index_type="HNSW"):
    """
    Executes the vector search.
    """
    # Force search to use COSINE to match the index
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}} # <--- CHANGED THIS from L2

    # Load collection just in case
    collection.load()

    results = collection.search(
        data=query_vectors,
        anns_field="vector",
        param=search_params,
        limit=limit,
        output_fields=["meta"]
    )
    return results[0]