#!/usr/bin/env python3
"""
SeRF Resume Script - Run only missing tasks
Reads existing CSV and runs only the missing (M, K, K_Search) combinations
"""

import subprocess
import os
import sys
import threading
import queue
import pandas as pd
from datetime import datetime
from pathlib import Path

# 禁用 Python 输出缓冲，实时显示
sys.stdout.reconfigure(line_buffering=True)

# ============== CONFIGURATION ==============

# Binary path
BINARY = "/home/djj/code/experiment/SeRF/build/benchmark/serf_arbitrary"

# Use existing CSV to determine missing tasks
EXISTING_CSV = "/home/djj/code/experiment/SeRF/results/serf_range_test/results_20260102_154738.csv"
OUTPUT_DIR = "/home/djj/code/experiment/SeRF/results/serf_range_test"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESUME_CSV = os.path.join(OUTPUT_DIR, f"resume_{TIMESTAMP}.csv")

# Leap strategy
STRATEGY = "MAX_POS"
STRATEGY_NAME = "MaxLeap"

# Dataset configurations
DATASETS = {
    "WIT-2048": "/home/djj/dataset/wit-image-random-1M.fvecs",
}

# Full parameter grid
M_VALUES = [8, 16, 32, 64]
K_VALUES = [100, 200, 400]
K_SEARCH_VALUES = [100, 200, 400]

# Execution settings
DATA_SIZE = 1000000
NUM_THREADS = 2
OMP_THREADS = 1

# ============== TASK GENERATION ==============

def get_missing_tasks():
    """Find missing tasks by comparing with existing CSV"""
    if not os.path.exists(EXISTING_CSV):
        print(f"WARNING: Existing CSV not found: {EXISTING_CSV}")x``
        print("Will run all tasks...")
        return None

    df = pd.read_csv(EXISTING_CSV)

    # Get completed (dataset, M, K, K_Search) combinations
    completed = set()
    for _, row in df.iterrows():
        completed.add((row['dataset'], row['M'], row['K'], row['K_Search']))

    print(f"Loaded existing CSV: {len(df)} records")
    print(f"Found {len(completed)} completed task combinations")

    # Generate all possible tasks and find missing ones
    all_tasks = []
    for dataset_name, dataset_path in DATASETS.items():
        for m in M_VALUES:
            for k in K_VALUES:
                for ks in K_SEARCH_VALUES:
                    key = (dataset_name, m, k, ks)
                    if key not in completed:
                        all_tasks.append({
                            'dataset': dataset_name,
                            'dataset_path': dataset_path,
                            'm': m,
                            'k': k,
                            'k_search': ks,
                            'data_size': DATA_SIZE,
                        })

    print(f"Missing tasks: {len(all_tasks)}")

    # Print missing tasks for verification
    if all_tasks:
        print("\nMissing tasks:")
        for t in all_tasks:
            print(f"  {t['dataset']}: M={t['m']:2d}, K={t['k']:3d}, KS={t['k_search']:3d}")
        print()

    return all_tasks

# ============== TASK EXECUTION ==============

def run_single_task(task, result_queue, lock):
    """Run a single serf_arbitrary test"""
    dataset_name = task['dataset']
    m = task['m']
    k = task['k']
    k_search = task['k_search']

    output_file = f"/tmp/serf_{dataset_name}_M{m}_K{k}_KS{k_search}.txt"

    # Set OMP threads for single-threaded execution
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS)

    # Build command
    cmd = [
        BINARY,
        "-dataset", "local",
        "-N", str(task['data_size']),
        "-dataset_path", task['dataset_path'],
        "-query_path", "",
        "-index_k", str(m),
        "-ef_con", str(k),
        "-ef_max", "500",
        "-ef_search", str(k_search),
    ]

    try:
        # Run the binary
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)

        # Parse build time
        build_time = None
        with open(output_file, 'r') as f:
            for line in f:
                if "Build Index Time" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        time_str = parts[4].rstrip('s')
                        try:
                            build_time = float(time_str)
                        except ValueError:
                            pass
                    break

        # Calculate IPS
        ips = None
        if build_time and build_time > 0:
            ips = task['data_size'] / build_time

        # Parse results (all ranges)
        results = []
        with open(output_file, 'r') as f:
            for line in f:
                if line.startswith("range:"):
                    parts = line.split()
                    if len(parts) >= 8:
                        range_val = int(parts[1])
                        recall = float(parts[3])
                        qps = float(parts[5])
                        comps = float(parts[7])

                        # Calculate range percentage
                        range_pct = (range_val * 100) // task['data_size']

                        results.append({
                            'dataset': dataset_name,
                            'leap_strategy': STRATEGY_NAME,
                            'param_type': 'All',
                            'M': m,
                            'K': k,
                            'K_Search': k_search,
                            'range_pct': range_pct,
                            'recall': recall,
                            'qps': qps,
                            'comps': comps,
                            'build_time': build_time,
                            'ips': ips,
                        })

        # Put results in queue
        with lock:
            for res in results:
                result_queue.put(res)

        # Print summary after putting results
        if results:
            res = results[0]
            print(f"  [OK] {dataset_name}: M={m:2d}, K={k:3d}, KS={k_search:3d} -> build={build_time:.2f}s, recall={res['recall']:.3f}")
        else:
            print(f"  [WARN] {dataset_name}: M={m}, K={k}, KS={k_search} -> No results parsed")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {dataset_name}: M={m}, K={k}, K_Search={k_search} - {e}")
    except Exception as e:
        print(f"[ERROR] {dataset_name}: M={m}, K={k}, K_Search={k_search} - {e}")
    finally:
        # Clean up temp file
        if os.path.exists(output_file):
            os.remove(output_file)

def worker(task_queue, result_queue, lock):
    """Worker thread function"""
    while True:
        try:
            task = task_queue.get_nowait()
            run_single_task(task, result_queue, lock)
            task_queue.task_done()
        except queue.Empty:
            break

# ============== MAIN ==============

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize CSV with header
    csv_header = "dataset,leap_strategy,param_type,M,K,K_Search,range_pct,recall,qps,comps,build_time,ips"

    # Get missing tasks
    tasks = get_missing_tasks()

    if not tasks:
        print("No missing tasks found!")
        return

    print("=" * 50)
    print("SeRF Resume Testing - Parallel Execution")
    print("=" * 50)
    print(f"Tasks to run: {len(tasks)}")
    print(f"Parallel Threads: {NUM_THREADS}")
    print(f"OMP Threads per Task: {OMP_THREADS}")
    print(f"Output CSV: {RESUME_CSV}")
    print("=" * 50)
    print()

    # Write CSV header
    with open(RESUME_CSV, 'w') as f:
        f.write(csv_header + '\n')

    # Create task queue
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    lock = threading.Lock()

    for task in tasks:
        task_queue.put(task)

    # Start worker threads
    threads = []
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(task_queue, result_queue, lock))
        t.start()
        threads.append(t)

    # Start result writer thread
    csv_lock = threading.Lock()
    completed_count = 0

    def result_writer():
        nonlocal completed_count
        retry_count = 0
        while True:
            try:
                result = result_queue.get(timeout=60)
                with csv_lock:
                    with open(RESUME_CSV, 'a') as f:
                        row = f"{result['dataset']},{result['leap_strategy']},{result['param_type']},{result['M']},{result['K']},{result['K_Search']},{result['range_pct']},{result['recall']},{result['qps']},{result['comps']},{result['build_time']},{result['ips']}\n"
                        f.write(row)
                completed_count += 1
                retry_count = 0
                if completed_count % 5 == 0:
                    print(f"  [Progress] {completed_count} results collected")
                result_queue.task_done()
            except queue.Empty:
                if all(not t.is_alive() for t in threads):
                    break
                else:
                    retry_count += 1
                    if retry_count <= 2:
                        print(f"  [Waiting] Workers still running... ({len([t for t in threads if t.is_alive()])} active)")
                    continue

    writer_thread = threading.Thread(target=result_writer)
    writer_thread.start()

    # Wait for all workers to complete
    for t in threads:
        t.join()

    # Wait for result writer to finish
    while not result_queue.empty():
        import time
        time.sleep(0.1)
    writer_thread.join()

    print()
    print("=" * 50)
    print(f"Resume testing completed!")
    print(f"Total records: {completed_count}")
    print(f"Results saved to: {RESUME_CSV}")
    print("=" * 50)

if __name__ == "__main__":
    main()
