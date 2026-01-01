#!/usr/bin/env python3
"""
Milvus Search Test - Single Dataset
Tests Recall and QPS for different query ranges
"""
import time
import sys
import numpy as np
import argparse
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

def load_fvecs(path, count=None):
    """Load fvecs file format"""
    with open(path, 'rb') as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        if count is None:
            data = np.frombuffer(f.read(), dtype=np.float32)
        else:
            bytes_to_read = int(count) * int(dim) * 4
            data = np.frombuffer(f.read(bytes_to_read), dtype=np.float32)
        return data.reshape(-1, dim)

def parse_index_params(s):
    parts = s.split(',')
    return {
        "metric_type": parts[1],
        "index_type": parts[0],
        "params": {"M": int(parts[2]), "efConstruction": int(parts[3])}
    }

def parse_search_params(s):
    parts = s.split(',')
    return {
        "metric_type": parts[0],
        "params": {"ef": int(parts[1])}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="19530")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--data-size", type=int, required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--index-params", required=True)
    parser.add_argument("--search-params", required=True)
    parser.add_argument("--range-pcts", nargs='+', type=int, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    index_params = parse_index_params(args.index_params)
    search_params = parse_search_params(args.search_params)

    print(f"========================================")
    print(f"Milvus Search Test: {args.dataset_name}")
    print(f"Collection: {args.collection_name}")
    print(f"Data Size: {args.data_size}, Dim: {args.dim}")
    print(f"========================================")

    # Connect
    connections.connect("default", host=args.host, port=args.port)

    # Check if collection exists
    if utility.has_collection(args.collection_name):
        print(f"Using existing collection: {args.collection_name}")
        collection = Collection(args.collection_name)
    else:
        print(f"Creating collection: {args.collection_name}")
        utility.drop_collection(args.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=args.dim)
        ]
        schema = CollectionSchema(fields, description="Search Test")
        collection = Collection(name=args.collection_name, schema=schema)

        # Insert data
        print(f"Loading data: {args.dataset_path}")
        vectors = load_fvecs(args.dataset_path, args.data_size)
        actual_size = vectors.shape[0]
        ids = np.arange(actual_size)

        # Batch size based on dimension
        if args.dim >= 2048:
            batch_size = 5000
        elif args.dim >= 960:
            batch_size = 10000
        else:
            batch_size = 20000

        for i in range(0, actual_size, batch_size):
            end = min(i + batch_size, actual_size)
            collection.insert([ids[i:end].tolist(), vectors[i:end]])
        collection.flush()

        # Build index
        print("Building index...")
        collection.create_index(field_name="embedding", index_params=index_params)

    # Load collection
    collection.load()

    # Get query vector
    query_data = load_fvecs(args.dataset_path, 1)
    query_vector = query_data[0:1].astype(np.float32)

    # Load vectors for brute force
    print("Loading vectors for ground truth...")
    vectors = load_fvecs(args.dataset_path, args.data_size)
    actual_size = vectors.shape[0]

    # Random start point for range (ensure range fits within data)
    # Use different random start for each range percentage
    np.random.seed(42)  # Fixed seed for reproducibility

    for range_pct in args.range_pcts:
        range_width = int(actual_size * range_pct / 100)
        # Random start: ensure [l_bound, l_bound + range_width] fits in [0, actual_size)
        max_l_bound = actual_size - range_width
        l_bound = np.random.randint(0, max(max_l_bound, 1))
        r_bound = l_bound + range_width - 1  # -1 because r_bound is inclusive
        expr = f"id >= {l_bound} && id <= {r_bound}"

        # Brute force ground truth
        range_vectors = vectors[l_bound:r_bound+1]
        distances = np.sum((query_vector - range_vectors) ** 2, axis=1)
        gt_indices_rel = np.argpartition(distances, args.top_k)[:args.top_k]
        gt_ids = set(l_bound + gt_indices_rel)

        # Warm up
        try:
            collection.search(query_vector, "embedding", search_params, args.top_k, expr=expr)
        except:
            pass

        # Measure
        t_start = time.time()
        results = []
        for _ in range(args.runs):
            res = collection.search(
                data=query_vector,
                anns_field="embedding",
                param=search_params,
                limit=args.top_k,
                expr=expr,
                output_fields=["id"]
            )
            results.append(res)
        t_end = time.time()

        avg_latency = (t_end - t_start) / args.runs * 1000
        qps = 1.0 / ((t_end - t_start) / args.runs)

        # Milvus returns SearchResult object - access hits differently
        result_ids = set(int(hit_id) for hit_id in results[0])
        recall = len(result_ids & gt_ids) / args.top_k if args.top_k > 0 else 0.0
        result_count = len(results[0])

        print(f"RANGE_{range_pct}PCT: LATENCY={avg_latency:.4f}ms QPS={qps:.2f} RECALL={recall:.4f} COUNT={result_count}")

    print("========================================")

if __name__ == "__main__":
    main()
