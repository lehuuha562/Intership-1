# reader.py
import argparse
from utils import load_json, show_search_results

def main():
    parser = argparse.ArgumentParser(description="Display search results.")
    parser.add_argument('--embeddings', default="data_embeddings.json", help='Path to embeddings JSON')
    parser.add_argument('--results', default="search_results.json", help='Path to search results JSON')
    parser.add_argument('--topk', type=int, default=5, help='Number of top results to display')
    args = parser.parse_args()
    
    embeddings = load_json(args.embeddings)
    results = load_json(args.results)
    
    if embeddings and results:
        show_search_results(results, embeddings, args.topk)
    else:
        print("Failed to load embeddings or results.")

if __name__ == "__main__":
    main()