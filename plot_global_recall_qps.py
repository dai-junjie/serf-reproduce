#!/usr/bin/env python3
"""
Plot Recall vs QPS for SeRF Global Range Benchmark
"""

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import numpy as np

# Global Range Search Data (1M Deep Dataset, 10000 queries)
global_data = {
    10: [(0.6972, 1136), (0.7910, 660), (0.8626, 380), (0.9188, 218)],
    20: [(0.8238, 1149), (0.9110, 676), (0.9588, 392), (0.9810, 226)],
    30: [(0.8347, 1151), (0.9164, 671), (0.9599, 398), (0.9810, 229)],
    50: [(0.8511, 1202), (0.9236, 700), (0.9635, 402), (0.9822, 231)],
}

# Bucket Search Data (buckets=20, for comparison)
bucket_data = {
    10: [(0.6581, 4002), (0.7407, 2290), (0.7924, 1303), (0.8197, 747)],
    20: [(0.7211, 4018), (0.8065, 2281), (0.8623, 1289), (0.8951, 739)],
    30: [(0.7656, 4078), (0.8477, 2335), (0.8984, 1328), (0.9282, 754)],
    50: [(0.7774, 4143), (0.8592, 2353), (0.9135, 1334), (0.9428, 756)],
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left plot: Global Range Search - different entry points
ax1 = axes[0]
for i, entry_points in enumerate([10, 20, 30, 50]):
    recalls = [d[0] for d in global_data[entry_points]]
    qps = [d[1] for d in global_data[entry_points]]
    ax1.plot(recalls, qps, marker=markers[i], color=colors[i],
            linestyle='-', linewidth=2.5, markersize=8, label=f'entry={entry_points}')

ax1.set_xlabel('Recall', fontsize=14)
ax1.set_ylabel('QPS', fontsize=14)
ax1.set_title('SeRF: Global Range Search (1M Dataset)', fontsize=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=12)
ax1.set_xlim(0.68, 0.99)
ax1.set_ylim(200, 1300)

# Right plot: Global vs Bucket (entry_points=30)
ax2 = axes[1]
entry_fixed = 30

# Global
recalls_g = [d[0] for d in global_data[entry_fixed]]
qps_g = [d[1] for d in global_data[entry_fixed]]
ax2.plot(recalls_g, qps_g, marker='o', color='#d62728',
        linestyle='-', linewidth=3, markersize=10, label='Global Range')

# Bucket (20 buckets)
recalls_b = [d[0] for d in bucket_data[entry_fixed]]
qps_b = [d[1] for d in bucket_data[entry_fixed]]
ax2.plot(recalls_b, qps_b, marker='s', color='#1f77b4',
        linestyle='--', linewidth=3, markersize=10, label='Bucket (20 buckets)')

ax2.set_xlabel('Recall', fontsize=14)
ax2.set_ylabel('QPS', fontsize=14)
ax2.set_title(f'SeRF: Global vs Bucket Search (entry_points={entry_fixed})', fontsize=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=12)
ax2.set_xlim(0.68, 0.99)
ax2.set_ylim(600, 4500)

plt.tight_layout()
plt.savefig('global_recall_qps_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: global_recall_qps_comparison.png")
plt.close()

# Single plot: All entry_points for Global Search
plt.figure(figsize=(10, 8))

for i, entry_points in enumerate([10, 20, 30, 50]):
    recalls = [d[0] for d in global_data[entry_points]]
    qps = [d[1] for d in global_data[entry_points]]
    plt.plot(recalls, qps, marker=markers[i], color=colors[i],
            linestyle='-', linewidth=2.5, markersize=9, label=f'entry_points={entry_points}')

plt.xlabel('Recall', fontsize=16)
plt.ylabel('QPS', fontsize=16)
plt.title('SeRF: Global Range Search - Recall vs QPS (1M Deep Dataset)', fontsize=18)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right', fontsize=14)
plt.xlim(0.68, 0.99)
plt.ylim(200, 1300)

plt.tight_layout()
plt.savefig('global_recall_qps_single.png', dpi=300, bbox_inches='tight')
print("Saved: global_recall_qps_single.png")
plt.close()

# Trade-off plot: speed vs accuracy
fig, ax = plt.subplots(figsize=(10, 7))

configs = [
    (10, 'entry=10 (fastest)', '#1f77b4', 'o'),
    (30, 'entry=30 (balanced)', '#ff7f0e', 's'),
    (50, 'entry=50 (most accurate)', '#d62728', '^'),
]

for entry_pts, label, color, marker in configs:
    recalls = [d[0] for d in global_data[entry_pts]]
    qps = [d[1] for d in global_data[entry_pts]]
    ax.plot(recalls, qps, marker=marker, color=color,
           linestyle='-', linewidth=3, markersize=11, label=label)

ax.set_xlabel('Recall', fontsize=16)
ax.set_ylabel('QPS', fontsize=16)
ax.set_title('SeRF Global: Speed-Accuracy Trade-off (1M Dataset)', fontsize=18)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=14)
ax.set_xlim(0.68, 0.99)
ax.set_ylim(200, 1300)

plt.tight_layout()
plt.savefig('global_recall_qps_tradeoff.png', dpi=300, bbox_inches='tight')
print("Saved: global_recall_qps_tradeoff.png")
plt.close()

# Data summary table
print("\n" + "="*70)
print("Global Range Search Results Summary (1M Deep Dataset)")
print("="*70)
print("| Entry Points | ef_search=128      | ef_search=256      | ef_search=512      | ef_search=1024     |")
print("-"*70)
for entry_points in [10, 20, 30, 50]:
    row = f"| {entry_points:12d}  | "
    for i, (recall, qps) in enumerate(global_data[entry_points]):
        row += f"R={recall:.4f} QPS={qps:4d}  "
    print(row + " |")
print("="*70)

print("\nComparison: Global vs Bucket (entry_points=30)")
print("-"*50)
print(f"{'ef_search':<12} | {'Global Recall':<15} | {'Bucket Recall':<15} | {'QPS Ratio':<10}")
print("-"*50)
for i, (rg, qg) in enumerate(global_data[30]):
    rb, qb = bucket_data[30][i]
    ratio = qb / qg
    print(f"{128*(2**i):<12} | {rg:<15.4f} | {rb:<15.4f} | {ratio:.2f}x")
print("-"*50)
