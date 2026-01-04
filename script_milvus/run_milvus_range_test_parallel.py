#!/usr/bin/env python3
"""
Parallel Milvus Range Testing
Runs parameter sweeps for multiple datasets with parallel execution
Datasets are processed sequentially (one at a time) for memory efficiency
Each dataset shares a single collection to reduce overhead
"""

import time
import sys
import os
import threading
import queue
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# 禁用 Python 输出缓冲，实时显示
sys.stdout.reconfigure(line_buffering=True)

# ============== CONFIGURATION ==============

# Milvus connection
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Output directory
OUTPUT_DIR = "/home/djj/code/experiment/SeRF/results/milvus_range_test"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
COMBINED_CSV = os.path.join(OUTPUT_DIR, f"results_{TIMESTAMP}.csv")

# Query ranges to test (as percentages) - including 0.1%
RANGE_PCTS = [0.1, 1, 10, 20, 50, 100]

# Dataset configurations: name:path (sorted by dimension)
DATASETS = {
    "DEEP-96": {
        "path": "/home/djj/code/experiment/timestampRAG/data/DEEP10M/deep_base.fvecs",
        "dim": 96,
    },
    "SIFT-128": {
        "path": "/home/djj/code/experiment/SeRF/data/sift_base.fvecs",
        "dim": 128,
    },
    "GIST-960": {
        "path": "/home/djj/code/experiment/timestampRAG/data/GIST1M/gist_base.fvecs",
        "dim": 960,
    },
    "WIT-2048": {
        "path": "/home/djj/dataset/wit-image-random-1M.fvecs",
        "dim": 2048,
    },
}

# Parameter grid for Milvus HNSW
# M: connectivity parameter (similar to index_k)
# EF_CONSTRUCTION: build time parameter (similar to ef_con/k)
# EF: search time parameter (similar to ef_search/k_search)
M_VALUES = [8, 16, 32, 64]
EF_CON_VALUES = [100, 200, 400]
EF_VALUES = [100, 200, 400]

# Execution settings
DATA_SIZE = 1000000
# Each dataset uses different thread count based on dimension
DATASET_THREADS = {
    "DEEP-96": 20,
    "SIFT-128": 16,
    "GIST-960": 10,
    "WIT-2048": 6,
}
RUNS_PER_TEST = 10  # Number of search runs for averaging
TOP_K = 10

# Milvus index type
INDEX_TYPE = "HNSW"
METRIC_TYPE = "L2"

# ============== UTILITY FUNCTIONS ==============

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

def get_batch_size(dim):
    """Get appropriate batch size based on dimension"""
    if dim >= 2048:
        return 5000
    elif dim >= 960:
        return 10000
    else:
        return 20000

# ============== TASK GENERATION ==============

def generate_tasks():
    """Generate all test tasks grouped by dataset"""
    tasks_by_dataset = {}

    for dataset_name, dataset_config in DATASETS.items():
        dataset_path = dataset_config["path"]
        if not os.path.exists(dataset_path):
            print(f"WARNING: Dataset not found: {dataset_path}")
            continue

        tasks_by_dataset[dataset_name] = []
        for m in M_VALUES:
            for ef_con in EF_CON_VALUES:
                for ef in EF_VALUES:
                    tasks_by_dataset[dataset_name].append({
                        'dataset': dataset_name,
                        'dataset_path': dataset_path,
                        'dim': dataset_config["dim"],
                        'm': m,
                        'ef_con': ef_con,
                        'ef': ef,
                        'data_size': DATA_SIZE,
                    })

    return tasks_by_dataset

# ============== MILVUS COLLECTION MANAGEMENT ==============

class MilvusCollectionManager:
    """Manages Milvus collection for a dataset - shared across all tasks"""

    def __init__(self, dataset_name, dataset_config, data_size):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_config["path"]
        self.dim = dataset_config["dim"]
        self.data_size = data_size
        self.collection = None
        self.base_collection_name = f"test_base_{dataset_name}"
        self.vectors_cache = None

    def setup(self):
        """Setup base collection with data"""
        # Connect to Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

        # Check if base collection exists
        if utility.has_collection(self.base_collection_name):
            print(f"  [Milvus] Using existing base collection: {self.base_collection_name}")
            self.collection = Collection(self.base_collection_name)
        else:
            print(f"  [Milvus] Creating base collection: {self.base_collection_name}")
            utility.drop_collection(self.base_collection_name)

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, description="Base Collection for Range Testing")
            self.collection = Collection(name=self.base_collection_name, schema=schema)

            # Load and insert data
            print(f"  [Milvus] Loading data: {self.dataset_path}")
            vectors = load_fvecs(self.dataset_path, self.data_size)
            actual_size = vectors.shape[0]
            ids = np.arange(actual_size)

            batch_size = get_batch_size(self.dim)
            print(f"  [Milvus] Inserting {actual_size} vectors (batch_size={batch_size})...")
            for i in range(0, actual_size, batch_size):
                end = min(i + batch_size, actual_size)
                self.collection.insert([ids[i:end].tolist(), vectors[i:end]])
            self.collection.flush()
            print(f"  [Milvus] Data inserted: {actual_size} vectors")

        return self.collection

    def get_test_collection(self, index_params):
        """Create or get a test collection with specific index"""
        test_collection_name = f"{self.base_collection_name}_M{index_params['params']['M']}_EFCON{index_params['params']['efConstruction']}"

        if utility.has_collection(test_collection_name):
            collection = Collection(test_collection_name)
            # Check if index exists
            indexes = collection.indexes
            if indexes:
                collection.load()
                return collection

        # Create new test collection from base
        print(f"  [Milvus] Creating test collection: {test_collection_name}")
        utility.drop_collection(test_collection_name)

        # Copy schema from base collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, description="Test Collection")
        collection = Collection(name=test_collection_name, schema=schema)

        # Load vectors from base collection and insert
        if self.vectors_cache is None:
            print(f"  [Milvus] Loading vectors for indexing: {self.dataset_path}")
            self.vectors_cache = load_fvecs(self.dataset_path, self.data_size)
            actual_size = self.vectors_cache.shape[0]
        else:
            actual_size = self.vectors_cache.shape[0]

        ids = np.arange(actual_size)
        batch_size = get_batch_size(self.dim)

        for i in range(0, actual_size, batch_size):
            end = min(i + batch_size, actual_size)
            collection.insert([ids[i:end].tolist(), self.vectors_cache[i:end]])
        collection.flush()

        # Build index
        print(f"  [Milvus] Building index: M={index_params['params']['M']}, efConstruction={index_params['params']['efConstruction']}")
        t0 = time.time()
        collection.create_index(field_name="embedding", index_params=index_params)
        build_time = time.time() - t0
        print(f"  [Milvus] Index built in {build_time:.2f}s")

        # Load collection
        collection.load()

        return collection, build_time

    def cleanup_test_collection(self, collection_name):
        """Cleanup a test collection after use"""
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
        except:
            pass

    def get_vectors_for_ground_truth(self):
        """Load vectors for ground truth computation"""
        if self.vectors_cache is None:
            print(f"  [Milvus] Loading vectors for ground truth: {self.dataset_path}")
            self.vectors_cache = load_fvecs(self.dataset_path, self.data_size)
        return self.vectors_cache

# ============== TASK EXECUTION ==============

def run_single_task(task, result_queue, lock, collection_manager):
    """Run a single Milvus test task"""
    dataset_name = task['dataset']
    m = task['m']
    ef_con = task['ef_con']
    ef = task['ef']

    # Build index params
    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": INDEX_TYPE,
        "params": {"M": m, "efConstruction": ef_con}
    }

    # Build search params
    search_params = {
        "metric_type": METRIC_TYPE,
        "params": {"ef": ef}
    }

    test_collection_name = f"{collection_manager.base_collection_name}_M{m}_EFCON{ef_con}"

    try:
        # Get or create test collection with index
        collection, build_time = collection_manager.get_test_collection(index_params)

        # Calculate IPS
        ips = None
        if build_time and build_time > 0:
            ips = task['data_size'] / build_time

        # Get query vector (use first vector as query)
        query_data = load_fvecs(task['dataset_path'], 1)
        query_vector = query_data[0:1].astype(np.float32)

        # Get vectors for ground truth
        vectors = collection_manager.get_vectors_for_ground_truth()
        actual_size = vectors.shape[0]

        # Fixed seed for reproducibility
        np.random.seed(42)

        results = []
        for range_pct in RANGE_PCTS:
            range_width = int(actual_size * range_pct / 100)
            # Ensure at least 1 element
            range_width = max(1, range_width)
            # Random start: ensure [l_bound, l_bound + range_width] fits in [0, actual_size)
            max_l_bound = actual_size - range_width
            l_bound = np.random.randint(0, max(max_l_bound, 1))
            r_bound = l_bound + range_width - 1
            expr = f"id >= {l_bound} && id <= {r_bound}"

            # Brute force ground truth
            range_vectors = vectors[l_bound:r_bound+1]
            distances = np.sum((query_vector - range_vectors) ** 2, axis=1)
            gt_indices_rel = np.argpartition(distances, TOP_K)[:TOP_K]
            gt_ids = set(l_bound + gt_indices_rel)

            # Warm up
            try:
                collection.search(query_vector, "embedding", search_params, TOP_K, expr=expr)
            except:
                pass

            # Measure search performance
            t_start = time.time()
            search_results = []
            for _ in range(RUNS_PER_TEST):
                res = collection.search(
                    data=query_vector,
                    anns_field="embedding",
                    param=search_params,
                    limit=TOP_K,
                    expr=expr,
                    output_fields=["id"]
                )
                search_results.append(res)
            t_end = time.time()

            avg_latency = (t_end - t_start) / RUNS_PER_TEST * 1000
            qps = 1.0 / ((t_end - t_start) / RUNS_PER_TEST)

            # Calculate recall
            result_ids = set(int(hit.id) for hit in search_results[0][0])
            recall = len(result_ids & gt_ids) / TOP_K if TOP_K > 0 else 0.0

            results.append({
                'dataset': dataset_name,
                'method': 'Milvus',
                'param_type': 'All',
                'M': m,
                'EF_Construction': ef_con,
                'EF': ef,
                'range_pct': range_pct,
                'recall': recall,
                'qps': qps,
                'comps': 0,  # Milvus doesn't expose distance computations
                'build_time': build_time,
                'ips': ips,
            })

        # Put results in queue
        with lock:
            for res in results:
                result_queue.put(res)

        # Print summary
        if results:
            res = results[0]
            print(f"  [OK] {dataset_name}: M={m:2d}, EFC={ef_con:3d}, EF={ef:3d} -> build={build_time:.2f}s, recall={res['recall']:.3f}")
        else:
            print(f"  [WARN] {dataset_name}: M={m}, EFC={ef_con}, EF={ef} -> No results")

    except Exception as e:
        print(f"[ERROR] {dataset_name}: M={m}, EFC={ef_con}, EF={ef} - {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup test collection to free memory
        try:
            collection_manager.cleanup_test_collection(test_collection_name)
        except:
            pass

def worker(task_queue, result_queue, lock, collection_manager):
    """Worker thread function"""
    while True:
        try:
            task = task_queue.get_nowait()
            run_single_task(task, result_queue, lock, collection_manager)
            task_queue.task_done()
        except queue.Empty:
            break

# ============== MAIN ==============

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize CSV with header
    csv_header = "dataset,method,param_type,M,EF_Construction,EF,range_pct,recall,qps,comps,build_time,ips"

    # Generate all tasks grouped by dataset
    tasks_by_dataset = generate_tasks()
    total_tasks = sum(len(tasks) for tasks in tasks_by_dataset.values())

    print("=" * 60)
    print("Milvus Range Testing - Parallel Execution")
    print("=" * 60)
    print(f"Datasets: {list(tasks_by_dataset.keys())}")
    print(f"Query Ranges: {RANGE_PCTS}%")
    print(f"Parameter Grid: M={M_VALUES}, EF_Construction={EF_CON_VALUES}, EF={EF_VALUES}")
    print(f"Total Tasks: {total_tasks}")
    print(f"Dataset Threads: {DATASET_THREADS}")
    print(f"Runs per Test: {RUNS_PER_TEST}")
    print(f"Output CSV: {COMBINED_CSV}")
    print("=" * 60)
    print()

    # Write CSV header
    with open(COMBINED_CSV, 'w') as f:
        f.write(csv_header + '\n')

    # Process datasets sequentially
    global_completed = 0
    csv_lock = threading.Lock()

    for dataset_idx, (dataset_name, dataset_tasks) in enumerate(tasks_by_dataset.items(), 1):
        print("=" * 60)
        print(f"[{dataset_idx}/{len(tasks_by_dataset)}] Processing Dataset: {dataset_name}")
        print("=" * 60)
        print(f"Tasks: {len(dataset_tasks)}")
        print(f"Data path: {DATASETS[dataset_name]['path']}")
        print(f"Dimension: {DATASETS[dataset_name]['dim']}")
        print("=" * 60)
        print()

        # Setup collection manager (shared base collection)
        collection_manager = MilvusCollectionManager(dataset_name, DATASETS[dataset_name], DATA_SIZE)
        collection_manager.setup()

        # Create task queue for this dataset
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        lock = threading.Lock()
        dataset_completed = 0

        for task in dataset_tasks:
            task_queue.put(task)

        def result_writer():
            nonlocal dataset_completed, global_completed
            retry_count = 0
            while True:
                try:
                    result = result_queue.get(timeout=60)
                    with csv_lock:
                        with open(COMBINED_CSV, 'a') as f:
                            row = f"{result['dataset']},{result['method']},{result['param_type']},{result['M']},{result['EF_Construction']},{result['EF']},{result['range_pct']},{result['recall']},{result['qps']},{result['comps']},{result['build_time']},{result['ips']}\n"
                            f.write(row)
                    dataset_completed += 1
                    global_completed += 1
                    retry_count = 0
                    if dataset_completed % 10 == 0:
                        print(f"  [Progress] {dataset_completed}/{len(dataset_tasks)} results collected (total: {global_completed}/{total_tasks})")
                    result_queue.task_done()
                except queue.Empty:
                    if all(not t.is_alive() for t in threads):
                        break
                    else:
                        retry_count += 1
                        if retry_count <= 2:
                            print(f"  [Waiting] Workers still running... ({len([t for t in threads if t.is_alive()])} active)")
                        continue

        # Start result writer thread
        writer_thread = threading.Thread(target=result_writer)
        writer_thread.start()

        # Get thread count for this dataset
        num_threads = DATASET_THREADS.get(dataset_name, 8)

        # Start worker threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(task_queue, result_queue, lock, collection_manager))
            t.start()
            threads.append(t)

        # Wait for all workers to complete
        for t in threads:
            t.join()

        # Wait for result writer to finish
        writer_thread.join()

        print(f"  [Done] Dataset {dataset_name} completed: {dataset_completed} results")
        print()

    print("=" * 60)
    print("All tests completed!")
    print(f"Total records: {global_completed}")
    print(f"Results saved to: {COMBINED_CSV}")
    print("=" * 60)

    # Generate plots
    print()
    print("=" * 60)
    print("Generating plots...")
    print("=" * 60)

    generate_plots(COMBINED_CSV, os.path.join(OUTPUT_DIR, f"plots_{TIMESTAMP}"))

    print()
    print("=" * 60)
    print("Complete!")
    print(f"  CSV: {COMBINED_CSV}")
    print(f"  Plots: {OUTPUT_DIR}/plots_{TIMESTAMP}/")
    print("=" * 60)

def generate_plots(csv_file, output_dir):
    """Generate visualization plots"""
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(csv_file)
    print(f"Data loaded: {len(df)} records")

    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12

    datasets = sorted(df['dataset'].unique())
    m_vals = sorted(df['M'].unique())
    ef_con_vals = sorted(df['EF_Construction'].unique())
    ef_vals = sorted(df['EF'].unique())
    range_pcts = sorted(df['range_pct'].unique())

    # For each dataset and each range_pct, create parameter heatmaps
    for dataset in datasets:
        df_ds = df[df['dataset'] == dataset]

        for rp in range_pcts:
            df_rp = df_ds[df_ds['range_pct'] == rp]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            for i, ef_val in enumerate(ef_vals):
                df_ef = df_rp[df_rp['EF'] == ef_val]
                if len(df_ef) > 0:
                    pivot_recall = df_ef.pivot(index='M', columns='EF_Construction', values='recall')
                    pivot_qps = df_ef.pivot(index='M', columns='EF_Construction', values='qps')

                    im1 = axes[0, i].imshow(pivot_recall.values, cmap='YlGnBu', aspect='auto')
                    axes[0, i].set_xticks(range(len(ef_con_vals)))
                    axes[0, i].set_xticklabels(ef_con_vals)
                    axes[0, i].set_yticks(range(len(m_vals)))
                    axes[0, i].set_yticklabels(m_vals)
                    axes[0, i].set_xlabel('EF_Construction')
                    axes[0, i].set_ylabel('M')
                    axes[0, i].set_title(f'Recall (EF={ef_val})')
                    plt.colorbar(im1, ax=axes[0, i])

                    for y in range(len(m_vals)):
                        for x in range(len(ef_con_vals)):
                            val = pivot_recall.values[y, x]
                            if not np.isnan(val):
                                axes[0, i].text(x, y, f'{val:.3f}', ha='center', va='center', fontsize=8)

                    im2 = axes[1, i].imshow(pivot_qps.values, cmap='YlOrRd', aspect='auto')
                    axes[1, i].set_xticks(range(len(ef_con_vals)))
                    axes[1, i].set_xticklabels(ef_con_vals)
                    axes[1, i].set_yticks(range(len(m_vals)))
                    axes[1, i].set_yticklabels(m_vals)
                    axes[1, i].set_xlabel('EF_Construction')
                    axes[1, i].set_ylabel('M')
                    axes[1, i].set_title(f'QPS (EF={ef_val})')
                    plt.colorbar(im2, ax=axes[1, i])

                    for y in range(len(m_vals)):
                        for x in range(len(ef_con_vals)):
                            val = pivot_qps.values[y, x]
                            if not np.isnan(val):
                                axes[1, i].text(x, y, f'{int(val)}', ha='center', va='center', fontsize=8)

            plt.suptitle(f'Milvus {dataset} - Parameter Sweep (Range {rp}%)', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/milvus_{dataset}_heatmap_range{rp}%.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_dir}/milvus_{dataset}_heatmap_range{rp}%.png")
            plt.close()

    # Line plots: Recall vs Range for different parameter values
    for dataset in datasets:
        df_ds = df[df['dataset'] == dataset]

        # Plot by M values
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for m in m_vals:
            df_m = df_ds[df_ds['M'] == m].groupby('range_pct').agg({'recall': 'mean', 'qps': 'mean'}).reset_index()
            axes[0].plot(df_m['range_pct'], df_m['recall'], marker='o', label=f'M={m}', linewidth=2)
            axes[1].plot(df_m['range_pct'], df_m['qps'], marker='s', label=f'M={m}', linewidth=2)

        axes[0].set_xlabel('Query Range (%)')
        axes[0].set_ylabel('Recall@10')
        axes[0].set_title(f'Milvus {dataset} - Recall vs Range (by M)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')

        axes[1].set_xlabel('Query Range (%)')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'Milvus {dataset} - QPS vs Range (by M)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/milvus_{dataset}_by_M.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/milvus_{dataset}_by_M.png")
        plt.close()

        # Plot by EF_Construction values
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ef_con in ef_con_vals:
            df_efc = df_ds[df_ds['EF_Construction'] == ef_con].groupby('range_pct').agg({'recall': 'mean', 'qps': 'mean'}).reset_index()
            axes[0].plot(df_efc['range_pct'], df_efc['recall'], marker='o', label=f'EF_Con={ef_con}', linewidth=2)
            axes[1].plot(df_efc['range_pct'], df_efc['qps'], marker='s', label=f'EF_Con={ef_con}', linewidth=2)

        axes[0].set_xlabel('Query Range (%)')
        axes[0].set_ylabel('Recall@10')
        axes[0].set_title(f'Milvus {dataset} - Recall vs Range (by EF_Construction)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')

        axes[1].set_xlabel('Query Range (%)')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'Milvus {dataset} - QPS vs Range (by EF_Construction)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/milvus_{dataset}_by_EF_Construction.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/milvus_{dataset}_by_EF_Construction.png")
        plt.close()

        # Plot by EF values
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ef_val in ef_vals:
            df_ef = df_ds[df_ds['EF'] == ef_val].groupby('range_pct').agg({'recall': 'mean', 'qps': 'mean'}).reset_index()
            axes[0].plot(df_ef['range_pct'], df_ef['recall'], marker='o', label=f'EF={ef_val}', linewidth=2)
            axes[1].plot(df_ef['range_pct'], df_ef['qps'], marker='s', label=f'EF={ef_val}', linewidth=2)

        axes[0].set_xlabel('Query Range (%)')
        axes[0].set_ylabel('Recall@10')
        axes[0].set_title(f'Milvus {dataset} - Recall vs Range (by EF)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')

        axes[1].set_xlabel('Query Range (%)')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'Milvus {dataset} - QPS vs Range (by EF)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/milvus_{dataset}_by_EF.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/milvus_{dataset}_by_EF.png")
        plt.close()

    print("All plots saved!")

if __name__ == "__main__":
    main()
