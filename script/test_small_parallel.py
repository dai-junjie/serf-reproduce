#!/usr/bin/env python3
"""
Small-scale test to verify parallel GT generation and parameter testing
Tests: 2 datasets × 3 ranges × 8 parameter combinations = 48 tasks
"""

import subprocess
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# ============== CONFIGURATION ==============

BINARY = "/home/djj/code/experiment/SeRF/build/benchmark/serf_arbitrary"
OUTPUT_DIR = "/home/djj/code/experiment/SeRF/results/test_small"
GROUNDTRUTH_DIR = "/home/djj/code/experiment/SeRF/results/groundtruth"
os.makedirs(GROUNDTRUTH_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Small test configuration
TEST_DATASETS = {
    "GIST-960": "/home/djj/dataset/gist_base.fvecs",
    "DEEP-96": "/home/djj/code/experiment/timestampRAG/data/DEEP10M/deep_base.fvecs",
}

DATA_SIZE = 100000    # Smaller for testing
QUERY_NUM = 100       # Fewer queries
RANGE_PCTS = [1, 10, 50]  # Only 3 ranges

M_VALUES = [8, 16]        # Only 2 M values
K_VALUES = [100, 200]     # Only 2 K values
K_SEARCH_VALUES = [100]   # Only 1 K_Search

NUM_THREADS = 8     # Fewer threads for testing
OMP_THREADS = 1
STRATEGY = "MAX_POS"
STRATEGY_NAME = "MaxLeap"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
COMBINED_CSV = os.path.join(OUTPUT_DIR, f"test_results_{TIMESTAMP}.csv")

# ============== GROUNDTRUTH GENERATION ==============

def generate_groundtruth_for_range(dataset_name, dataset_path, range_pct):
    """Generate groundtruth for ONE specific range"""
    gt_file = os.path.join(GROUNDTRUTH_DIR, f"{dataset_name}_N{DATA_SIZE}_Q{QUERY_NUM}_R{range_pct}_groundtruth.csv")

    if os.path.exists(gt_file):
        print(f"  [SKIP] GT exists: {dataset_name} R{range_pct}%")
        return gt_file

    cmd = [
        BINARY, "-dataset", "local", "-N", str(DATA_SIZE),
        "-dataset_path", dataset_path, "-query_path", "",
        "-groundtruth_path", gt_file, "-generate_gt_only",
    ]

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        print(f"  [DONE] GT generated: {dataset_name} R{range_pct}%")
        return gt_file
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] GT failed for {dataset_name} R{range_pct}%: {e.stderr}")
        return None

def pregenerate_all_groundtruth():
    """Generate groundtruth for all dataset-range combinations IN PARALLEL"""
    print("=" * 50)
    print("Step 1: Pre-generating Groundtruth (Parallel)")
    print("=" * 50)

    gt_tasks = []
    for dataset_name, dataset_path in TEST_DATASETS.items():
        if not os.path.exists(dataset_path):
            print(f"  [SKIP] Dataset not found: {dataset_path}")
            continue
        for range_pct in RANGE_PCTS:
            gt_file = os.path.join(GROUNDTRUTH_DIR, f"{dataset_name}_N{DATA_SIZE}_Q{QUERY_NUM}_R{range_pct}_groundtruth.csv")
            if not os.path.exists(gt_file):
                gt_tasks.append((dataset_name, dataset_path, range_pct))

    if not gt_tasks:
        print("  [SKIP] All groundtruth files already exist!")
        return

    print(f"  Total tasks: {len(gt_tasks)}")
    print(f"  Parallel threads: {NUM_THREADS}")
    print()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(generate_groundtruth_for_range, ds, path, rng): (ds, rng)
                   for ds, path, rng in gt_tasks}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 2 == 0:
                print(f"  [Progress] {completed}/{len(gt_tasks)} GT files generated")

    print("\nGroundtruth generation complete!\n")

# ============== TASK EXECUTION ==============

def run_single_task(task, result_queue, lock):
    """Run a single serf_arbitrary test"""
    dataset_name = task['dataset']
    range_pct = task['range_pct']
    m, k, k_search = task['m'], task['k'], task['k_search']

    output_file = f"/tmp/serf_test_{dataset_name}_R{range_pct}_M{m}_K{k}_KS{k_search}.txt"
    gt_file = task['gt_file']

    if not os.path.exists(gt_file):
        print(f"[ERROR] No GT file for {dataset_name} R{range_pct}%")
        return

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS)

    cmd = [
        BINARY, "-dataset", "local", "-N", str(task['data_size']),
        "-dataset_path", task['dataset_path'], "-query_path", "",
        "-groundtruth_path", gt_file, "-index_k", str(m),
        "-ef_con", str(k), "-ef_max", "500", "-ef_search", str(k_search),
        "-recursion_type", STRATEGY,
    ]

    try:
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)

        # Parse results
        build_time = None
        with open(output_file, 'r') as f:
            for line in f:
                if "Build Index Time" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            build_time = float(parts[4].rstrip('s'))
                        except ValueError:
                            pass
                    break

        ips = build_time and build_time > 0 and task['data_size'] / build_time or None

        results = []
        with open(output_file, 'r') as f:
            for line in f:
                if line.startswith("range:"):
                    parts = line.split()
                    if len(parts) >= 8:
                        range_val = int(parts[1])
                        calc_pct = (range_val * 100) // task['data_size']
                        if calc_pct == range_pct:
                            results.append({
                                'dataset': dataset_name,
                                'leap_strategy': STRATEGY_NAME,
                                'param_type': 'All',
                                'M': m, 'K': k, 'K_Search': k_search,
                                'range_pct': range_pct,
                                'recall': float(parts[3]),
                                'qps': float(parts[5]),
                                'comps': float(parts[7]),
                                'build_time': build_time,
                                'ips': ips,
                            })

        with lock:
            for res in results:
                result_queue.put(res)

        if results:
            res = results[0]
            print(f"[OK] {dataset_name} R{range_pct}%: M={m}, K={k}, KS={k_search} -> recall={res['recall']:.3f}, qps={res['qps']:.0f}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {dataset_name} R{range_pct}%: M={m}, K={k}, KS={k_search} - {e}")
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
    csv_header = "dataset,leap_strategy,param_type,M,K,K_Search,range_pct,recall,qps,comps,build_time,ips"

    # Step 1: Generate GT
    pregenerate_all_groundtruth()

    # Collect GT files
    gt_files_map = {}
    for dataset_name, dataset_path in TEST_DATASETS.items():
        if not os.path.exists(dataset_path):
            continue
        gt_files_map[dataset_name] = {}
        for range_pct in RANGE_PCTS:
            gt_file = os.path.join(GROUNDTRUTH_DIR, f"{dataset_name}_N{DATA_SIZE}_Q{QUERY_NUM}_R{range_pct}_groundtruth.csv")
            if os.path.exists(gt_file):
                gt_files_map[dataset_name][range_pct] = gt_file

    total_gt = sum(len(r) for r in gt_files_map.values())
    print(f"Using {total_gt} GT files ({len(gt_files_map)} datasets × {len(RANGE_PCTS)} ranges)\n")

    # Write CSV header
    with open(COMBINED_CSV, 'w') as f:
        f.write(csv_header + '\n')

    # Step 2: Run tests
    print("=" * 50)
    print("Step 2: Running Parameter Tests (Dataset by Dataset)")
    print("=" * 50)

    all_results = []

    for dataset_name in gt_files_map.keys():
        dataset_path = TEST_DATASETS[dataset_name]
        dataset_gt_files = gt_files_map[dataset_name]

        # Generate tasks
        dataset_params = []
        for m in M_VALUES:
            for k in K_VALUES:
                for k_search in K_SEARCH_VALUES:
                    for range_pct in RANGE_PCTS:
                        if range_pct in dataset_gt_files:
                            dataset_params.append({
                                'dataset': dataset_name,
                                'dataset_path': dataset_path,
                                'range_pct': range_pct,
                                'gt_file': dataset_gt_files[range_pct],
                                'm': m, 'k': k, 'k_search': k_search,
                                'data_size': DATA_SIZE,
                            })

        print(f"\nDataset {dataset_name}: {len(dataset_params)} tasks")
        print(f"  Params: {len(M_VALUES)}×{len(K_VALUES)}×{len(K_SEARCH_VALUES)} = {len(M_VALUES)*len(K_VALUES)*len(K_SEARCH_VALUES)}")
        print(f"  Ranges: {len(dataset_gt_files)}")
        print(f"  Total: {len(dataset_params)} tasks")
        print()

        # Run batch
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        lock = threading.Lock()

        for task in dataset_params:
            task_queue.put(task)

        threads = []
        for _ in range(NUM_THREADS):
            t = threading.Thread(target=worker, args=(task_queue, result_queue, lock))
            t.start()
            threads.append(t)

        completed = 0
        batch_results = []

        def result_collector():
            nonlocal completed
            while True:
                try:
                    result = result_queue.get(timeout=1)
                    batch_results.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  [Progress] {completed}/{len(dataset_params)} tasks completed")
                    result_queue.task_done()
                except queue.Empty:
                    break

        collector = threading.Thread(target=result_collector)
        collector.start()

        for t in threads:
            t.join()

        while not result_queue.empty():
            import time
            time.sleep(0.1)
        collector.join()

        all_results.extend(batch_results)

        # Write results
        with open(COMBINED_CSV, 'a') as f:
            for res in batch_results:
                row = f"{res['dataset']},{res['leap_strategy']},{res['param_type']},{res['M']},{res['K']},{res['K_Search']},{res['range_pct']},{res['recall']},{res['qps']},{res['comps']},{res['build_time']},{res['ips']}\n"
                f.write(row)

        print(f"[DONE] {dataset_name}: {completed} results\n")

    print("=" * 50)
    print(f"Test Complete! Total: {len(all_results)} results")
    print(f"CSV: {COMBINED_CSV}")
    print("=" * 50)

if __name__ == "__main__":
    main()
