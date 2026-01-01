#!/usr/bin/env python3
"""
Parallel SeRF Range Testing
Runs parameter sweeps for multiple datasets with parallel execution
"""

import subprocess
import os
import threading
import queue
import pandas as pd
from datetime import datetime
from pathlib import Path

# ============== CONFIGURATION ==============

# Binary path
BINARY = "/home/djj/code/experiment/SeRF/build/benchmark/serf_arbitrary"

# Output directory
OUTPUT_DIR = "/home/djj/code/experiment/SeRF/results/serf_range_test"
GROUNDTRUTH_DIR = "/home/djj/code/experiment/SeRF/results/groundtruth"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
COMBINED_CSV = os.path.join(OUTPUT_DIR, f"results_{TIMESTAMP}.csv")

# Create directories
os.makedirs(GROUNDTRUTH_DIR, exist_ok=True)

# Query ranges to test (as percentages)
RANGE_PCTS = [1, 10, 20, 50, 100]

# Leap strategy
STRATEGY = "MAX_POS"
STRATEGY_NAME = "MaxLeap"

# Dataset configurations: name:path (sorted by dimension)
DATASETS = {
    # "DEEP-96": "/home/djj/code/experiment/timestampRAG/data/DEEP10M/deep_base.fvecs",
    "SIFT-128": "/home/djj/code/experiment/SeRF/data/sift_base.fvecs",
    "GIST-960": "/home/djj/code/experiment/timestampRAG/data/GIST1M/gist_base.fvecs",
    "WIT-2048": "/home/djj/dataset/wit-image-random-1M.fvecs",
}

# Parameter grid
M_VALUES = [8, 16, 32, 64]
K_VALUES = [100, 200, 400]
K_SEARCH_VALUES = [100, 200, 400]

# Execution settings
DATA_SIZE = 1000000
QUERY_NUM = 1000    # Number of queries for groundtruth generation
GT_THREADS = 10     # Threads for GT generation (lighter memory usage)
NUM_THREADS = 30    # Threads for parameter testing
OMP_THREADS = 1     # Threads per task (single-threaded)

# ============== TASK GENERATION ==============

def generate_tasks():
    """Generate all test tasks"""
    tasks = []

    for dataset_name, dataset_path in DATASETS.items():
        if not os.path.exists(dataset_path):
            print(f"WARNING: Dataset not found: {dataset_path}")
            continue

        for m in M_VALUES:
            for k in K_VALUES:
                for k_search in K_SEARCH_VALUES:
                    tasks.append({
                        'dataset': dataset_name,
                        'dataset_path': dataset_path,
                        'm': m,
                        'k': k,
                        'k_search': k_search,
                        'data_size': DATA_SIZE,
                    })

    return tasks

# ============== GROUNDTRUTH GENERATION ==============

def generate_groundtruth_for_range(dataset_name, dataset_path, range_pct):
    """
    Generate groundtruth for ONE specific range.
    GT file is named with range percentage: {dataset}_N{N}_Q{Q}_R{range_pct}_groundtruth.csv
    """
    gt_file = os.path.join(GROUNDTRUTH_DIR, f"{dataset_name}_N{DATA_SIZE}_Q{QUERY_NUM}_R{range_pct}_groundtruth.csv")

    # Check if already exists
    if os.path.exists(gt_file):
        print(f"  [SKIP] GT exists: {dataset_name} R{range_pct}%")
        return gt_file

    print(f"  [START] Generating GT: {dataset_name} R{range_pct}%...")

    # Run serf_arbitrary with -generate_gt_only to generate groundtruth WITHOUT building index
    cmd = [
        BINARY,
        "-dataset", "local",
        "-N", str(DATA_SIZE),
        "-dataset_path", dataset_path,
        "-query_path", "",
        "-groundtruth_path", gt_file,  # Save to this file
        "-generate_gt_only",  # Don't build index, just generate GT
    ]

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_THREADS)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        print(f"  [DONE] GT generated: {dataset_name} R{range_pct}%")
        return gt_file
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Failed to generate GT for {dataset_name} R{range_pct}%")
        print(f"    stderr: {e.stderr}")
        return None

def pregenerate_all_groundtruth():
    """Generate groundtruth for all dataset-range combinations IN PARALLEL"""
    print("=" * 50)
    print("Step 1: Pre-generating Groundtruth (Parallel)")
    print("=" * 50)

    # Create a list of all GT generation tasks: (dataset_name, dataset_path, range_pct)
    gt_tasks = []
    for dataset_name, dataset_path in DATASETS.items():
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

    print(f"  Total tasks: {len(gt_tasks)} (datasets × ranges)")
    print(f"  Parallel threads: {GT_THREADS}")
    print()

    # Use ThreadPoolExecutor for parallel GT generation
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=GT_THREADS) as executor:
        futures = {
            executor.submit(generate_groundtruth_for_range, ds_name, ds_path, rng): (ds_name, rng)
            for ds_name, ds_path, rng in gt_tasks
        }

        completed = 0
        for future in as_completed(futures):
            ds_name, rng = futures[future]
            completed += 1
            # Print progress more frequently for better feedback
            if completed % 2 == 0 or completed == len(gt_tasks):
                print(f"  [Progress] {completed}/{len(gt_tasks)} GT files generated")

    print()
    print("=" * 50)
    print("Groundtruth generation complete!")
    print("=" * 50 + "\n")

# ============== TASK EXECUTION ==============

def run_single_task(task, result_queue, lock):
    """Run a single serf_arbitrary test for ONE dataset + ONE range + ONE parameter set"""
    dataset_name = task['dataset']
    range_pct = task['range_pct']
    m = task['m']
    k = task['k']
    k_search = task['k_search']

    output_file = f"/tmp/serf_{dataset_name}_R{range_pct}_M{m}_K{k}_KS{k_search}.txt"

    # GT file is specific to this dataset and range
    gt_file = task['gt_file']
    if not gt_file or not os.path.exists(gt_file):
        print(f"[ERROR] No GT file for {dataset_name} R{range_pct}%")
        return

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
        "-groundtruth_path", gt_file,
        "-index_k", str(m),
        "-ef_con", str(k),
        "-ef_max", "500",
        "-ef_search", str(k_search),
        "-recursion_type", STRATEGY,
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

        # Parse results - should only have ONE range result
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

                        # Verify range matches expected
                        calc_range_pct = (range_val * 100) // task['data_size']
                        if calc_range_pct == range_pct:
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

        if results:
            res = results[0]
            print(f"[OK] {dataset_name} R{range_pct}%: M={m}, K={k}, KS={k_search} -> recall={res['recall']:.3f}, qps={res['qps']:.0f}")
        else:
            print(f"[WARN] {dataset_name} R{range_pct}%: M={m}, K={k}, KS={k_search} -> No results parsed")

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

def run_batch_for_dataset(dataset_name, dataset_path, all_params):
    """
    Run all parameter+range combinations for a single dataset.
    Tasks include GT file path specific to each range.
    """
    print("=" * 50)
    print(f"Processing Dataset: {dataset_name}")
    print("=" * 50)
    print(f"Base dataset: {dataset_path}")
    print(f"Total tasks: {len(all_params)}")
    print(f"Parallel threads: {NUM_THREADS}")
    print("=" * 50)
    print()

    # Create task queue and result queue
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    lock = threading.Lock()

    for task in all_params:
        task_queue.put(task)

    # Start worker threads
    threads = []
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(task_queue, result_queue, lock))
        t.start()
        threads.append(t)

    # Monitor progress and collect results
    completed = 0
    batch_results = []
    total_tasks = len(all_params)

    # Start a thread to collect results
    def result_collector():
        nonlocal completed
        while True:
            try:
                result = result_queue.get(timeout=1)
                batch_results.append(result)
                completed += 1
                if completed % 20 == 0:
                    print(f"[Progress] {completed}/{total_tasks} tasks completed for {dataset_name}")
                result_queue.task_done()
            except queue.Empty:
                # Only exit if all workers are done
                if all(not t.is_alive() for t in threads):
                    break

    collector_thread = threading.Thread(target=result_collector)
    collector_thread.start()

    # Wait for all workers to complete
    for t in threads:
        t.join()

    # Wait for result collector to finish
    while not result_queue.empty():
        import time
        time.sleep(0.1)
    collector_thread.join()

    print()
    print(f"[DONE] Dataset {dataset_name} completed: {completed} results")
    print()

    return batch_results

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize CSV with header
    csv_header = "dataset,leap_strategy,param_type,M,K,K_Search,range_pct,recall,qps,comps,build_time,ips"

    # Step 1: Pre-generate groundtruth for all dataset-range combinations IN PARALLEL
    pregenerate_all_groundtruth()

    # Collect all groundtruth file paths: {dataset_name: {range_pct: gt_file}}
    gt_files_map = {}
    for dataset_name, dataset_path in DATASETS.items():
        if not os.path.exists(dataset_path):
            continue
        gt_files_map[dataset_name] = {}
        for range_pct in RANGE_PCTS:
            gt_file = os.path.join(GROUNDTRUTH_DIR, f"{dataset_name}_N{DATA_SIZE}_Q{QUERY_NUM}_R{range_pct}_groundtruth.csv")
            if os.path.exists(gt_file):
                gt_files_map[dataset_name][range_pct] = gt_file

    if len(gt_files_map) == 0:
        print("ERROR: No groundtruth files generated!")
        return

    # Count total GT files
    total_gt_files = sum(len(ranges) for ranges in gt_files_map.values())
    print(f"Using {total_gt_files} groundtruth files ({len(gt_files_map)} datasets × {len(RANGE_PCTS)} ranges)")
    print()

    # Write CSV header
    with open(COMBINED_CSV, 'w') as f:
        f.write(csv_header + '\n')

    # Step 2: Process each dataset SEPARATELY with ALL ranges
    print("=" * 50)
    print("Step 2: Running Parameter Sweep Tests (Dataset by Dataset)")
    print("=" * 50)
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Query Ranges: {RANGE_PCTS}%")
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Parameter Grid: M={M_VALUES}, K={K_VALUES}, K_Search={K_SEARCH_VALUES}")
    print(f"Parallel Threads per Dataset: {NUM_THREADS}")
    print(f"OMP Threads per Task: {OMP_THREADS}")
    print(f"Output CSV: {COMBINED_CSV}")
    print("=" * 50)
    print()

    # Process each dataset one by one
    all_results = []

    for dataset_name in gt_files_map.keys():
        dataset_path = DATASETS[dataset_name]
        dataset_gt_files = gt_files_map[dataset_name]

        # Generate all tasks: parameters × ranges
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
                                'm': m,
                                'k': k,
                                'k_search': k_search,
                                'data_size': DATA_SIZE,
                            })

        print(f"Dataset {dataset_name}: {len(dataset_params)} tasks ({len(M_VALUES)*len(K_VALUES)*len(K_SEARCH_VALUES)} params × {len(dataset_gt_files)} ranges)")

        # Run batch for this dataset
        batch_results = run_batch_for_dataset(dataset_name, dataset_path, dataset_params)
        all_results.extend(batch_results)

        # Write results for this dataset to CSV immediately
        with open(COMBINED_CSV, 'a') as f:
            for result in batch_results:
                row = f"{result['dataset']},{result['leap_strategy']},{result['param_type']},{result['M']},{result['K']},{result['K_Search']},{result['range_pct']},{result['recall']},{result['qps']},{result['comps']},{result['build_time']},{result['ips']}\n"
                f.write(row)

    print()
    print("=" * 50)
    print("All datasets completed!")
    print(f"Total records: {len(all_results)}")
    print(f"Results saved to: {COMBINED_CSV}")
    print("=" * 50)

    # Generate plots
    print()
    print("=" * 50)
    print("Generating plots...")
    print("=" * 50)

    generate_plots(COMBINED_CSV, os.path.join(OUTPUT_DIR, f"plots_{TIMESTAMP}"))

    print()
    print("=" * 50)
    print("Complete!")
    print(f"  CSV: {COMBINED_CSV}")
    print(f"  Plots: {OUTPUT_DIR}/plots_{TIMESTAMP}/")
    print("=" * 50)

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
    k_vals = sorted(df['K'].unique())
    ks_vals = sorted(df['K_Search'].unique())
    range_pcts = sorted(df['range_pct'].unique())

    # For each dataset and each range_pct, create parameter heatmaps
    for dataset in datasets:
        df_ds = df[df['dataset'] == dataset]

        for rp in range_pcts:
            df_rp = df_ds[df_ds['range_pct'] == rp]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            for i, ks in enumerate(ks_vals):
                df_ks = df_rp[df_rp['K_Search'] == ks]
                if len(df_ks) > 0:
                    pivot_recall = df_ks.pivot(index='M', columns='K', values='recall')
                    pivot_qps = df_ks.pivot(index='M', columns='K', values='qps')

                    im1 = axes[0, i].imshow(pivot_recall.values, cmap='YlGnBu', aspect='auto')
                    axes[0, i].set_xticks(range(len(k_vals)))
                    axes[0, i].set_xticklabels(k_vals)
                    axes[0, i].set_yticks(range(len(m_vals)))
                    axes[0, i].set_yticklabels(m_vals)
                    axes[0, i].set_xlabel('K')
                    axes[0, i].set_ylabel('M')
                    axes[0, i].set_title(f'Recall (K_Search={ks})')
                    plt.colorbar(im1, ax=axes[0, i])

                    for y in range(len(m_vals)):
                        for x in range(len(k_vals)):
                            val = pivot_recall.values[y, x]
                            axes[0, i].text(x, y, f'{val:.3f}', ha='center', va='center', fontsize=8)

                    im2 = axes[1, i].imshow(pivot_qps.values, cmap='YlOrRd', aspect='auto')
                    axes[1, i].set_xticks(range(len(k_vals)))
                    axes[1, i].set_xticklabels(k_vals)
                    axes[1, i].set_yticks(range(len(m_vals)))
                    axes[1, i].set_yticklabels(m_vals)
                    axes[1, i].set_xlabel('K')
                    axes[1, i].set_ylabel('M')
                    axes[1, i].set_title(f'QPS (K_Search={ks})')
                    plt.colorbar(im2, ax=axes[1, i])

                    for y in range(len(m_vals)):
                        for x in range(len(k_vals)):
                            val = pivot_qps.values[y, x]
                            axes[1, i].text(x, y, f'{int(val)}', ha='center', va='center', fontsize=8)

            plt.suptitle(f'{dataset} - Parameter Sweep (Range {int(rp)}%)', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{dataset}_heatmap_range{int(rp)}%.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_dir}/{dataset}_heatmap_range{int(rp)}%.png")
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
        axes[0].set_title(f'{dataset} - Recall vs Range (by M)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Query Range (%)')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'{dataset} - QPS vs Range (by M)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{dataset}_by_M.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/{dataset}_by_M.png")
        plt.close()

        # Plot by K values
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for k in k_vals:
            df_k = df_ds[df_ds['K'] == k].groupby('range_pct').agg({'recall': 'mean', 'qps': 'mean'}).reset_index()
            axes[0].plot(df_k['range_pct'], df_k['recall'], marker='o', label=f'K={k}', linewidth=2)
            axes[1].plot(df_k['range_pct'], df_k['qps'], marker='s', label=f'K={k}', linewidth=2)

        axes[0].set_xlabel('Query Range (%)')
        axes[0].set_ylabel('Recall@10')
        axes[0].set_title(f'{dataset} - Recall vs Range (by K)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Query Range (%)')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'{dataset} - QPS vs Range (by K)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{dataset}_by_K.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/{dataset}_by_K.png")
        plt.close()

        # Plot by K_Search values
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ks in ks_vals:
            df_ks = df_ds[df_ds['K_Search'] == ks].groupby('range_pct').agg({'recall': 'mean', 'qps': 'mean'}).reset_index()
            axes[0].plot(df_ks['range_pct'], df_ks['recall'], marker='o', label=f'K_Search={ks}', linewidth=2)
            axes[1].plot(df_ks['range_pct'], df_ks['qps'], marker='s', label=f'K_Search={ks}', linewidth=2)

        axes[0].set_xlabel('Query Range (%)')
        axes[0].set_ylabel('Recall@10')
        axes[0].set_title(f'{dataset} - Recall vs Range (by K_Search)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Query Range (%)')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'{dataset} - QPS vs Range (by K_Search)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{dataset}_by_K_Search.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/{dataset}_by_K_Search.png")
        plt.close()

    print("All plots saved!")

if __name__ == "__main__":
    main()
