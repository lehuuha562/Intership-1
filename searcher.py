# searcher.py
from pymilvus import (
    Collection,
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
)
from rich.console import Console

console = Console()

VECTOR_DIM = 512
DEFAULT_COLLECTION = "demo_embeddings"


def setup_collection(
    collection_name: str = DEFAULT_COLLECTION,
    dim: int = VECTOR_DIM,
    index_type: str = "HNSW",
):
    """
    Create or load a Milvus collection with a selectable index type.

    index_type:
        - "FLAT"  : Exact cosine search (ground truth)
        - "HNSW"  : Approximate nearest neighbor search
    """

    if not connections.has_connection("default"):
        raise ConnectionError("No active connection to Milvus.")

    # If collection already exists, reuse it
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
            ),
            FieldSchema(
                name="meta",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
        ]

        schema = CollectionSchema(
            fields, description="Multimodal Vector Search Collection"
        )
        collection = Collection(collection_name, schema)

    # Drop existing index if switching index types
    # Drop existing index safely
    if collection.has_index():
        console.print("[yellow]Releasing collection before dropping index...[/yellow]")
        try:
            collection.release()
        except Exception:
            pass  # already released

    console.print("[yellow]Dropping existing index...[/yellow]")
    collection.drop_index()

    # Index configuration
    if index_type == "FLAT":
        index_params = {
            "index_type": "FLAT",
            "metric_type": "COSINE",
            "params": {},
        }

    elif index_type == "HNSW":
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 8,
                "efConstruction": 64,
            },
        }

    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    console.print(
        f"[cyan]Building index: {index_type} (COSINE)...[/cyan]"
    )
    collection.create_index(
        field_name="vector",
        index_params=index_params,
    )
    console.print("[green]Index ready.[/green]")

    return collection


def perform_search(
    collection: Collection,
    query_vectors,
    limit: int = 5,
    index_type: str = "HNSW",
):
    """
    Execute vector search using the specified index type.
    """

    if index_type == "FLAT":
        search_params = {
            "metric_type": "COSINE",
            "params": {},
        }

    elif index_type == "HNSW":
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "ef": 64,
            },
        }

    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    collection.load()

    results = collection.search(
        data=query_vectors,
        anns_field="vector",
        param=search_params,
        limit=limit,
        output_fields=["meta"],
    )

    # Return hits directly (Milvus returns a list per query)
    return results[0]