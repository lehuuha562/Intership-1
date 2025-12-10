# pipeline.py
import sys
import os
import argparse
import numpy as np
from embedder import embed_text
from searcher import perform_search
from utils import prepare_collection, console
from rich.table import Table

# Default fallback path if no arguments are provided
DEFAULT_DATA_PATH = "data/books.csv"

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Multimodal Search Pipeline")
    parser.add_argument(
        "--path", 
        type=str, 
        default=DEFAULT_DATA_PATH, 
        help="Path to a CSV file (for books) OR a Directory (for images/video/audio)"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="fantasy novels about dragons", 
        help="The search query text"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data_path = args.path
    
    console.print(f"[bold cyan]Target Data Path:[/bold cyan] {data_path}")

    # Step 1: Initialize Collection (Autodetects CSV vs Folder)
    try:
        collection = prepare_collection(data_path)
    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        sys.exit(1)

    # Step 2: Embed query
    console.print(f"\n[bold cyan]Query:[/bold cyan] '{args.query}'")
    
    query_vec = [embed_text(args.query)]
    if query_vec[0] is None or np.linalg.norm(query_vec[0]) == 0:
        console.print("[red]Error: Invalid query vector[/red]")
        sys.exit(1)

    # Step 3: Search
    results = perform_search(collection, query_vec, limit=5, index_type="IVF_FLAT")
    
    console.print("\n[bold]Search Results:[/bold]")
    if results:
        table = Table(title=f"Top Matches")
        table.add_column("Rank", style="cyan", width=5)
        table.add_column("Score", style="green", width=10)
        table.add_column("Type / Meta", style="white")
        
        for rank, hit in enumerate(results, 1):
            # Parse the meta field to display it nicely
            meta = hit["meta"]
            
            # Simple formatter to highlight if it's an image, sound, or book
            if " | " in meta:
                type_tag, content = meta.split(" | ", 1)
                display_str = f"[yellow][{type_tag}][/yellow] {content}"
            else:
                display_str = meta

            table.add_row(str(rank), f"{hit['distance']:.4f}", display_str)
        console.print(table)
    else:
        console.print("[red]No search results returned[/red]")
    
    console.print(f"\n[dim]Total entities in collection: {collection.num_entities}[/dim]")