#!/usr/bin/env python3
"""
Plot WIT-Image parameter test results
 Generates 3 groups of plots:
  1. M parameter: QPS vs M, Recall vs M
  2. K parameter: QPS vs K, Recall vs K
  3. K_Search parameter: QPS vs K_Search, Recall vs K_Search
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (12, 4)

# Read CSV
csv_file = 'results_20251229_135128.csv'
df = pd.read_csv(csv_file)

print("Data loaded:")
print(df)
print()

# Create output directory
import os
os.makedirs('plots', exist_ok=True)

# ===========================
# Plot 1: M parameter
# ===========================
df_M = df[df['param_type'] == 'M'].copy()
if len(df_M) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1.1: Recall vs M
    axes[0].plot(df_M['M'], df_M['recall'], marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
    axes[0].set_xlabel('M (index_k)')
    axes[0].set_ylabel('Recall@10%')
    axes[0].set_title('Recall vs M')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([min(df_M['recall']) * 0.99, 1.0])

    # Plot 1.2: QPS vs M
    ax2 = axes[1].plot(df_M['M'], df_M['qps'], marker='s', linestyle='-', color='coral', linewidth=2, markersize=8)
    axes[1].set_xlabel('M (index_k)')
    axes[1].set_ylabel('QPS')
    axes[1].set_title('QPS vs M')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/M_parameter.png', dpi=150, bbox_inches='tight')
    print("Saved: plots/M_parameter.png")
    plt.close()

# ===========================
# Plot 2: K parameter
# ===========================
df_K = df[df['param_type'] == 'K'].copy()
if len(df_K) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 2.1: Recall vs K
    axes[0].plot(df_K['K'], df_K['recall'], marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
    axes[0].set_xlabel('K (ef_construction)')
    axes[0].set_ylabel('Recall@10%')
    axes[0].set_title('Recall vs K')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([min(df_K['recall']) * 0.99, 1.0])

    # Plot 2.2: QPS vs K
    axes[1].plot(df_K['K'], df_K['qps'], marker='s', linestyle='-', color='coral', linewidth=2, markersize=8)
    axes[1].set_xlabel('K (ef_construction)')
    axes[1].set_ylabel('QPS')
    axes[1].set_title('QPS vs K')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/K_parameter.png', dpi=150, bbox_inches='tight')
    print("Saved: plots/K_parameter.png")
    plt.close()

# ===========================
# Plot 3: K_Search parameter
# ===========================
df_KS = df[df['param_type'] == 'K_Search'].copy()
if len(df_KS) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 3.1: Recall vs K_Search
    axes[0].plot(df_KS['K_Search'], df_KS['recall'], marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
    axes[0].set_xlabel('K_Search (ef_search)')
    axes[0].set_ylabel('Recall@10%')
    axes[0].set_title('Recall vs K_Search')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([min(df_KS['recall']) * 0.99, 1.0])

    # Plot 3.2: QPS vs K_Search
    axes[1].plot(df_KS['K_Search'], df_KS['qps'], marker='s', linestyle='-', color='coral', linewidth=2, markersize=8)
    axes[1].set_xlabel('K_Search (ef_search)')
    axes[1].set_ylabel('QPS')
    axes[1].set_title('QPS vs K_Search')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/K_Search_parameter.png', dpi=150, bbox_inches='tight')
    print("Saved: plots/K_Search_parameter.png")
    plt.close()

# ===========================
# Combined plot: All three parameters in one figure
# ===========================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Row 1: Recall plots
if len(df_M) > 0:
    axes[0, 0].plot(df_M['M'], df_M['recall'], marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('M')
    axes[0, 0].set_ylabel('Recall@10%')
    axes[0, 0].set_title('Recall vs M')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.85, 1.0])

if len(df_K) > 0:
    axes[0, 1].plot(df_K['K'], df_K['recall'], marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('K')
    axes[0, 1].set_ylabel('Recall@10%')
    axes[0, 1].set_title('Recall vs K')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.85, 1.0])

if len(df_KS) > 0:
    axes[0, 2].plot(df_KS['K_Search'], df_KS['recall'], marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('K_Search')
    axes[0, 2].set_ylabel('Recall@10%')
    axes[0, 2].set_title('Recall vs K_Search')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0.85, 1.0])

# Row 2: QPS plots
if len(df_M) > 0:
    axes[1, 0].plot(df_M['M'], df_M['qps'], marker='s', linestyle='-', color='coral', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('M')
    axes[1, 0].set_ylabel('QPS')
    axes[1, 0].set_title('QPS vs M')
    axes[1, 0].grid(True, alpha=0.3)

if len(df_K) > 0:
    axes[1, 1].plot(df_K['K'], df_K['qps'], marker='s', linestyle='-', color='coral', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('K')
    axes[1, 1].set_ylabel('QPS')
    axes[1, 1].set_title('QPS vs K')
    axes[1, 1].grid(True, alpha=0.3)

if len(df_KS) > 0:
    axes[1, 2].plot(df_KS['K_Search'], df_KS['qps'], marker='s', linestyle='-', color='coral', linewidth=2, markersize=8)
    axes[1, 2].set_xlabel('K_Search')
    axes[1, 2].set_ylabel('QPS')
    axes[1, 2].set_title('QPS vs K_Search')
    axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/all_parameters_combined.png', dpi=150, bbox_inches='tight')
print("Saved: plots/all_parameters_combined.png")
plt.close()

print("\nAll plots saved in 'plots/' directory")
