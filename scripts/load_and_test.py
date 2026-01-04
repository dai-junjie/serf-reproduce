#!/usr/bin/env python3
"""
SeRF/HNSW Load and Test Script
Load existing indexes and run tests with multiple K_Search values
"""

import subprocess
import os
import sys
import threading
import queue
import pandas as pd
from datetime import datetime
import glob

# ============== CONFIGURATION ==============

# Binary paths
SERF_BINARY = "/home/djj/code/experiment/SeRF/build/benchmark/serf_save_load"
HNSW_BINARY = "/home/djj/code/experiment/SeRF/build/benchmark/hnsw_save_load"

# Index directory
INDEX_DIR = "/home/djj/code/experiment/SeRF/results/indexes"

# Query ranges to test (as percentages)
RANGE_PCTS = [0.1, 1, 10, 20, 50, 100]

# Leap strategy (for SeRF)
STRATEGY = "MAX_POS"

# Dataset path map
DATASETS = {
    "DEEP-96": "/home/djj/code/experiment/timestampRAG/data/DEEP10M/deep_base.fvecs",
    "SIFT-128": "/home/djj/code/experiment/SeRF/data/sift_base.fvecs",
    "GIST-960": "/home/djj/code/experiment/timestampRAG/data/GIST1M/gist_base.fvecs",
    "WIT-2048": "/home/djj/dataset/wit-image-random-1M.fvecs",
}

# K_Search values to test (for each loaded index)
K_SEARCH_VALUES = [100, 200, 300, 400]

# Execution settings
DATA_SIZE = 1000000
DATASET_THREADS = {
    "DEEP-96": 24,
    "SIFT-128": 16,
    "GIST-960": 10,
    "WIT-2048": 6,
}
OMP_THREADS = 1

# Method names (for display only)
METHOD_NAMES = {
    "serf": "SeRF",
    "hnsw": "HNSW",
}

# ============== INDEX DISCOVERY ==============

def parse_index_filename(filename):
    """Parse index filename to extract parameters (compatible with old and new format)"""
    basename = os.path.basename(filename)
    name = basename.replace('.bin', '')
    parts = name.split('_')

    # New format: method_dataset1m_M_K (4 parts)
    # Old format: method_dataset1m_M_K_KS (5 parts)
    if len(parts) < 4:
        return None

    method = parts[0]
    if method not in ["serf", "hnsw"]:
        return None

    try:
        dataset_part = parts[1]
        dataset_base = dataset_part.replace('1m', '').upper()
        dataset_name = None
        for full_name in DATASETS.keys():
            if full_name.startswith(dataset_base):
                dataset_name = full_name
                break
        if dataset_name is None:
            return None
        m = int(parts[2].replace('M', ''))
        k = int(parts[3].replace('K', ''))
        # Ignore KS if present (old format)
        return {
            'method': method,
            'dataset': dataset_name,
            'm': m,
            'k': k,
            'index_path': filename,
        }
    except (ValueError, IndexError):
        return None

def discover_load_tasks():
    """Discover all existing indexes"""
    index_files = glob.glob(os.path.join(INDEX_DIR, "*.bin"))
    tasks_by_dataset = {}
    for index_file in index_files:
        params = parse_index_filename(index_file)
        if params is None:
            continue
        dataset_name = params['dataset']
        if dataset_name not in DATASETS:
            continue
        if dataset_name not in tasks_by_dataset:
            tasks_by_dataset[dataset_name] = []
        if not os.path.exists(DATASETS[dataset_name]):
            continue
        tasks_by_dataset[dataset_name].append({
            'method': params['method'],
            'dataset': dataset_name,
            'dataset_path': DATASETS[dataset_name],
            'm': params['m'],
            'k': params['k'],
            'k_search_list': K_SEARCH_VALUES,  # Test with multiple K_Search values
            'data_size': DATA_SIZE,
            'index_path': index_file,
        })
    return tasks_by_dataset

# ============== TASK EXECUTION ==============

def get_binary(method):
    return SERF_BINARY if method == "serf" else HNSW_BINARY

def run_single_task(task, result_queue, lock):
    """Run a single load+test task with multiple K_Search values"""
    method = task['method']
    dataset_name = task['dataset']
    m = task['m']
    k = task['k']
    k_search_list = task['k_search_list']
    index_path = task['index_path']

    method_name = METHOD_NAMES[method]
    output_file = f"/tmp/{method}_{dataset_name}_M{m}_K{k}_load.txt"

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS)

    cmd = [
        get_binary(method),
        "-dataset", "local",
        "-N", str(task['data_size']),
        "-dataset_path", task['dataset_path'],
        "-query_path", "",
        "-index_k", str(m),
        "-ef_con", str(k),
        "-ef_max", "500",
        "-ef_search_list", ",".join(map(str, k_search_list)),  # Test multiple K_Search values
        "-load_index", index_path,
    ]

    # Add SeRF-specific parameters
    if method == "serf":
        cmd.extend(["-recursion_type", STRATEGY])

    try:
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)

        # Parse load time
        load_time = None
        with open(output_file, 'r') as f:
            for line in f:
                if "Load" in line and "Time" in line and "Index" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        time_str = parts[-1].rstrip('s')
                        try:
                            load_time = float(time_str)
                        except ValueError:
                            pass
                    break

        # Parse results - now with multiple K_Search values
        results = []
        current_k_search = None

        with open(output_file, 'r') as f:
            for line in f:
                # Detect which K_Search we're currently testing
                if "Testing with search_ef=" in line:
                    current_k_search = int(line.split("=")[1].strip().split()[0])
                    continue

                if line.startswith("range:"):
                    parts = line.split()
                    if len(parts) >= 8:
                        range_val = int(parts[1])
                        recall = float(parts[3])
                        qps = float(parts[5])
                        comps = float(parts[7])
                        range_pct = (range_val * 100) // task['data_size']

                        results.append({
                            'method': method,
                            'dataset': dataset_name,
                            'M': m,
                            'K': k,
                            'K_Search': current_k_search,
                            'range_pct': range_pct,
                            'recall': recall,
                            'qps': qps,
                            'comps': comps,
                            'load_time': load_time,
                        })

        with lock:
            for res in results:
                result_queue.put(res)

        if results:
            if load_time is not None:
                load_str = f"load={load_time:.4f}s"
            else:
                load_str = "load=N/A"
            ks_str = ",".join(map(str, k_search_list))
            print(f"  [OK] {method_name:8} {dataset_name}: M={m:2d}, K={k:3d}, KS=[{ks_str}] -> {load_str} ({len(results)} results)")
        else:
            ks_str = ",".join(map(str, k_search_list))
            print(f"  [WARN] {method_name:8} {dataset_name}: M={m}, K={k}, KS=[{ks_str}] -> No results")

    except Exception as e:
        ks_str = ",".join(map(str, k_search_list))
        print(f"[ERROR] {method_name:8} {dataset_name}: M={m}, K={k}, KS=[{ks_str}] - {e}")
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)

def worker(task_queue, result_queue, lock):
    while True:
        try:
            task = task_queue.get_nowait()
            run_single_task(task, result_queue, lock)
            task_queue.task_done()
        except queue.Empty:
            break

# ============== MAIN ==============

def main():
    # Output directory
    OUTPUT_DIR = "/home/djj/code/experiment/SeRF/results/load_test"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover available indexes
    tasks_by_dataset = discover_load_tasks()

    if not tasks_by_dataset:
        print("=" * 60)
        print("ERROR: No indexes found!")
        print(f"Please run build_and_test.py first to build indexes.")
        print(f"Index directory: {INDEX_DIR}")
        print("=" * 60)
        return

    total_tasks = sum(len(tasks) for tasks in tasks_by_dataset.values())

    print("=" * 60)
    print("SeRF/HNSW Load Test - Parallel Execution")
    print("=" * 60)
    print(f"Methods: SeRF, HNSW")
    print(f"Total Indexes Found: {total_tasks}")
    print(f"K_Search values to test: {K_SEARCH_VALUES}")
    print(f"Index Dir: {INDEX_DIR}")
    print("=" * 60)
    print()

    # CSV header
    csv_header = "method,dataset,M,K,K_Search,range_pct,recall,qps,comps,load_time"
    COMBINED_CSV = os.path.join(OUTPUT_DIR, f"load_test_{TIMESTAMP}.csv")

    with open(COMBINED_CSV, 'w') as f:
        f.write(csv_header + '\n')

    global_completed = 0

    # Process each dataset
    for dataset_idx, (dataset_name, dataset_tasks) in enumerate(tasks_by_dataset.items(), 1):
        print("=" * 60)
        print(f"[{dataset_idx}/{len(tasks_by_dataset)}] {dataset_name}")
        print("=" * 60)
        print(f"Indexes to test: {len(dataset_tasks)}")
        print("=" * 60)
        print()

        task_queue = queue.Queue()
        result_queue = queue.Queue()
        lock = threading.Lock()
        dataset_completed = 0

        for task in dataset_tasks:
            task_queue.put(task)

        csv_lock = threading.Lock()
        workers_done = threading.Event()

        def result_writer():
            nonlocal dataset_completed, global_completed
            while True:
                try:
                    result = result_queue.get(timeout=2)
                    with csv_lock:
                        with open(COMBINED_CSV, 'a') as f:
                            row = f"{result['method']},{result['dataset']},{result['M']},{result['K']},{result['K_Search']},{result['range_pct']},{result['recall']},{result['qps']},{result['comps']},{result['load_time']}\n"
                            f.write(row)
                    dataset_completed += 1
                    global_completed += 1
                    if dataset_completed % 10 == 0:
                        print(f"  [Progress] {dataset_completed}/{len(dataset_tasks)} results")
                    result_queue.task_done()
                except queue.Empty:
                    if workers_done.is_set() and result_queue.empty():
                        break
                    continue

        writer_thread = threading.Thread(target=result_writer)
        writer_thread.start()

        threads = []
        num_threads = DATASET_THREADS.get(dataset_name, 8)
        for _ in range(num_threads):
            t = threading.Thread(target=worker, args=(task_queue, result_queue, lock))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        workers_done.set()
        writer_thread.join(timeout=10)

        print(f"  [Done] {dataset_completed} results")
        print()

    print("=" * 60)
    print("All tests completed!")
    print(f"Total records: {global_completed}")
    print(f"Results saved to: {COMBINED_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()
