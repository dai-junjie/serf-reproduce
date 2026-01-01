#!/usr/bin/env python3
"""
Parallel HNSW Range Testing
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
BINARY = "/home/djj/code/experiment/SeRF/build/benchmark/benchmark_hnsw_arbitrary"

# Output directory
OUTPUT_DIR = "/home/djj/code/experiment/SeRF/results/hnsw_range_test"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
COMBINED_CSV = os.path.join(OUTPUT_DIR, f"results_{TIMESTAMP}.csv")

# Query ranges to test (as percentages)
RANGE_PCTS = [1, 10, 20, 50, 100]

# Dataset configurations: name:path (sorted by dimension)
DATASETS = {
    "DEEP-96": "/home/djj/code/experiment/timestampRAG/data/DEEP10M/deep_base.fvecs",
    "SIFT-128": "/home/djj/code/experiment/SeRF/data/sift_base.fvecs",
    "GIST-960": "/home/djj/code/experiment/timestampRAG/data/GIST1M/gist_base.fvecs",
    "WIT-2048": "/home/djj/dataset/wit-image-random-1M.fvecs",
}

# Parameter grid (same as SeRF)
M_VALUES = [8, 16, 32, 64]
K_VALUES = [100, 200, 400]
K_SEARCH_VALUES = [100, 200, 400]

# Execution settings
DATA_SIZE = 1000000
NUM_THREADS = 28  # Number of parallel tasks
OMP_THREADS = 1   # Threads per task (single-threaded)

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

# ============== TASK EXECUTION ==============

def run_single_task(task, result_queue, lock):
    """Run a single benchmark_hnsw_arbitrary test"""
    dataset_name = task['dataset']
    m = task['m']
    k = task['k']
    k_search = task['k_search']

    output_file = f"/tmp/hnsw_{dataset_name}_M{m}_K{k}_KS{k_search}.txt"

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

        # Parse build time first
        build_time = None
        with open(output_file, 'r') as f:
            for line in f:
                if "Build Index Time" in line:
                    # Format: "# Build Index Time: 98.8173730s"
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

        # Parse results
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

                        if range_pct in RANGE_PCTS:
                            results.append({
                                'dataset': dataset_name,
                                'method': 'HNSW',
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

        print(f"[OK] {dataset_name}: M={m}, K={k}, K_Search={k_search} -> {len(results)} results (build_time={build_time:.2f}s, ips={ips:.0f})")

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
    csv_header = "dataset,method,param_type,M,K,K_Search,range_pct,recall,qps,comps,build_time,ips"

    # Generate all tasks
    tasks = generate_tasks()
    total_tasks = len(tasks)

    print("=" * 50)
    print("HNSW Baseline Range Testing - Parallel Execution")
    print("=" * 50)
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Query Ranges: {RANGE_PCTS}%")
    print(f"Parameter Grid: M={M_VALUES}, K={K_VALUES}, K_Search={K_SEARCH_VALUES}")
    print(f"Total Tasks: {total_tasks}")
    print(f"Parallel Threads: {NUM_THREADS}")
    print(f"OMP Threads per Task: {OMP_THREADS}")
    print(f"Output CSV: {COMBINED_CSV}")
    print("=" * 50)
    print()

    # Create task queue
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    lock = threading.Lock()

    for task in tasks:
        task_queue.put(task)

    # Write CSV header
    with open(COMBINED_CSV, 'w') as f:
        f.write(csv_header + '\n')

    # Start worker threads
    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(task_queue, result_queue, lock))
        t.start()
        threads.append(t)

    # Monitor progress
    completed = 0

    # Start a thread to write results to CSV incrementally
    csv_lock = threading.Lock()

    def result_writer():
        while True:
            try:
                result = result_queue.get(timeout=1)
                with csv_lock:
                    with open(COMBINED_CSV, 'a') as f:
                        row = f"{result['dataset']},{result['method']},{result['param_type']},{result['M']},{result['K']},{result['K_Search']},{result['range_pct']},{result['recall']},{result['qps']},{result['comps']},{result['build_time']},{result['ips']}\n"
                        f.write(row)
                nonlocal completed
                completed += 1
                if completed % 10 == 0:
                    print(f"[Progress] {completed}/{total_tasks} tasks completed")
                result_queue.task_done()
            except queue.Empty:
                break

    # Start result writer thread
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
    print("All tests completed!")
    print(f"Total records: {completed}")
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

            plt.suptitle(f'HNSW {dataset} - Parameter Sweep (Range {int(rp)}%)', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/hnsw_{dataset}_heatmap_range{int(rp)}%.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_dir}/hnsw_{dataset}_heatmap_range{int(rp)}%.png")
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
        axes[0].set_title(f'HNSW {dataset} - Recall vs Range (by M)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Query Range (%)')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'HNSW {dataset} - QPS vs Range (by M)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/hnsw_{dataset}_by_M.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/hnsw_{dataset}_by_M.png")
        plt.close()

        # Plot by K values
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for k in k_vals:
            df_k = df_ds[df_ds['K'] == k].groupby('range_pct').agg({'recall': 'mean', 'qps': 'mean'}).reset_index()
            axes[0].plot(df_k['range_pct'], df_k['recall'], marker='o', label=f'K={k}', linewidth=2)
            axes[1].plot(df_k['range_pct'], df_k['qps'], marker='s', label=f'K={k}', linewidth=2)

        axes[0].set_xlabel('Query Range (%)')
        axes[0].set_ylabel('Recall@10')
        axes[0].set_title(f'HNSW {dataset} - Recall vs Range (by K)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Query Range (%)')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'HNSW {dataset} - QPS vs Range (by K)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/hnsw_{dataset}_by_K.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/hnsw_{dataset}_by_K.png")
        plt.close()

        # Plot by K_Search values
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ks in ks_vals:
            df_ks = df_ds[df_ds['K_Search'] == ks].groupby('range_pct').agg({'recall': 'mean', 'qps': 'mean'}).reset_index()
            axes[0].plot(df_ks['range_pct'], df_ks['recall'], marker='o', label=f'K_Search={ks}', linewidth=2)
            axes[1].plot(df_ks['range_pct'], df_ks['qps'], marker='s', label=f'K_Search={ks}', linewidth=2)

        axes[0].set_xlabel('Query Range (%)')
        axes[0].set_ylabel('Recall@10')
        axes[0].set_title(f'HNSW {dataset} - Recall vs Range (by K_Search)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Query Range (%)')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'HNSW {dataset} - QPS vs Range (by K_Search)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/hnsw_{dataset}_by_K_Search.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/hnsw_{dataset}_by_K_Search.png")
        plt.close()

    print("All plots saved!")

if __name__ == "__main__":
    main()
