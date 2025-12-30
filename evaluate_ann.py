import time
import numpy as np
from embedder import embed_text
from searcher import setup_collection, perform_search
from utils import prepare_collection, connect_milvus

TOP_K = 10
NUM_QUERIES = 20

queries = [
    "books about dragons",
    "space exploration science fiction",
    "romantic novels",
    "stories about artificial intelligence",
    "fantasy magic worlds",
    "historical war stories",
    "detective mystery novels",
    "philosophy of mind",
    "adventure in the jungle",
    "mythical creatures",
] * 2  # duplicate to reach 20


def recall_at_k(exact, ann, k):
    return len(set(exact[:k]) & set(ann[:k])) / k

def evaluate():
    connect_milvus()

    collection = prepare_collection("data/books.csv")

    # ---------- Exact Search ----------
    exact_col = setup_collection(index_type="FLAT")
    exact_col.load()

    # ---------- ANN Search ----------
    ann_col = setup_collection(index_type="HNSW")
    ann_col.load()

    recalls = []
    exact_times = []
    ann_times = []

    for q in queries:
        qvec = [embed_text(q)]

        # Exact
        t0 = time.time()
        exact_res = perform_search(
            exact_col, qvec, limit=TOP_K, index_type="FLAT"
        )
        exact_times.append(time.time() - t0)

        exact_metas = [hit.entity.get("meta") for hit in exact_res]

        # ANN
        t0 = time.time()
        ann_res = perform_search(
            ann_col, qvec, limit=TOP_K, index_type="HNSW"
        )
        ann_times.append(time.time() - t0)

        ann_metas   = [hit.entity.get("meta") for hit in ann_res]

        recalls.append(recall_at_k(exact_metas, ann_metas, TOP_K))

    print("\n=== Evaluation Results ===")
    print(f"Recall@{TOP_K}: {np.mean(recalls):.4f}")
    print(f"Exact Avg Latency: {np.mean(exact_times)*1000:.2f} ms")
    print(f"HNSW Avg Latency: {np.mean(ann_times)*1000:.2f} ms")


if __name__ == "__main__":
    evaluate()
