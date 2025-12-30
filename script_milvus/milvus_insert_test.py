#!/usr/bin/env python3
"""
Milvus Insert Test - Single Configuration
Tests IPS (Insert Per Second) based on index build time
"""
import time
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
    args = parser.parse_args()

    index_params = parse_index_params(args.index_params)

    print(f"========================================")
    print(f"Milvus Insert Test: {args.dataset_name}")
    print(f"Collection: {args.collection_name}")
    print(f"Data Size: {args.data_size}, Dim: {args.dim}")
    print(f"========================================")

    # Connect
    connections.connect("default", host=args.host, port=args.port)

    # Create collection
    if utility.has_collection(args.collection_name):
        utility.drop_collection(args.collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=args.dim)
    ]
    schema = CollectionSchema(fields, description="Insert Test")
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

    t_start = time.time()
    for i in range(0, actual_size, batch_size):
        end = min(i + batch_size, actual_size)
        collection.insert([ids[i:end].tolist(), vectors[i:end]])
    collection.flush()
    insert_time = time.time() - t_start
    print(f"INSERT_TIME: {insert_time:.4f}")

    # Build index
    print("Building index...")
    t0 = time.time()
    collection.create_index(field_name="embedding", index_params=index_params)
    build_time = time.time() - t0
    print(f"BUILD_TIME: {build_time:.4f}")

    # IPS based on build time (HNSW insertion time = index build time)
    ips = actual_size / build_time
    print(f"IPS: {ips:.2f}")

    print("========================================")

if __name__ == "__main__":
    main()
