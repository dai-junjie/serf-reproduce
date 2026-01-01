#!/usr/bin/env python3
"""
Quick test: DEEP dataset with N=1000, Q=100
Test all 5 range percentages (1%, 10%, 20%, 50%, 100%)
"""

import subprocess
import os

# Config
BINARY = "/home/djj/code/experiment/SeRF/build/benchmark/serf_arbitrary"
DATASET_NAME = "DEEP-96"
DATASET_PATH = "/home/djj/code/experiment/timestampRAG/data/DEEP10M/deep_base.fvecs"
DATA_SIZE = 10000
QUERY_NUM = 100

# Parameters to test
M = 8
K = 100
K_SEARCH = 400
STRATEGY = "MAX_POS"

# Output
OUTPUT_FILE = "/tmp/deep_test_N10000_Q100.txt"

print("=" * 60)
print("Quick Test: DEEP Dataset")
print("=" * 60)
print(f"Dataset: {DATASET_NAME}")
print(f"Data Size: {DATA_SIZE}")
print(f"Query Num: {QUERY_NUM}")
print(f"Parameters: M={M}, K={K}, K_Search={K_SEARCH}")
print(f"Strategy: {STRATEGY}")
print(f"Ranges: 1%, 10%, 20%, 50%, 100%")
print("=" * 60)
print()

# Step 1: Generate groundtruth
print("Step 1: Generating groundtruth...")
gt_file = f"/tmp/{DATASET_NAME}_N{DATA_SIZE}_Q{QUERY_NUM}_groundtruth.csv"

cmd_gt = [
    BINARY,
    "-dataset", "local",
    "-N", str(DATA_SIZE),
    "-dataset_path", DATASET_PATH,
    "-query_path", "",
    "-groundtruth_path", gt_file,
    "-generate_gt_only",
]

print(f"Running: {' '.join(cmd_gt)}")
result = subprocess.run(cmd_gt, capture_output=True, text=True)
if result.returncode != 0:
    print(f"ERROR: Groundtruth generation failed!")
    print(result.stderr)
    exit(1)
print(f"Groundtruth saved to: {gt_file}")
print()

# Step 2: Run test with cached groundtruth
print("Step 2: Running test with cached groundtruth...")
cmd_test = [
    BINARY,
    "-dataset", "local",
    "-N", str(DATA_SIZE),
    "-dataset_path", DATASET_PATH,
    "-query_path", "",
    "-groundtruth_path", gt_file,
    "-index_k", str(M),
    "-ef_con", str(K),
    "-ef_max", "500",
    "-ef_search", str(K_SEARCH),
    "-recursion_type", STRATEGY,
]

print(f"Running: {' '.join(cmd_test)}")
print()

result = subprocess.run(cmd_test, capture_output=True, text=True)

# Save output
with open(OUTPUT_FILE, 'w') as f:
    f.write(result.stdout)
    if result.stderr:
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

print(f"Output saved to: {OUTPUT_FILE}")
print()

# Show results
print("=" * 60)
print("Results (range by range):")
print("=" * 60)

# Parse and show the range results
import re
for line in result.stdout.split('\n'):
    if line.startswith('range:'):
        print(line)

print()
print("=" * 60)
print("Test Complete!")
print("=" * 60)
