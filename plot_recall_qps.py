#!/usr/bin/env python3
"""
Plot Recall vs QPS for SeRF Bucket Benchmark
Cleaner version with strategic data selection
"""

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np

# Data from benchmark results (1M dataset)
# Format: (recall, qps) for each (buckets, entry_points, ef_search) configuration

data = {
    20: {  # 20 buckets
        10: [(0.6581, 4002), (0.7407, 2290), (0.7924, 1303), (0.8197, 747)],
        20: [(0.7211, 4018), (0.8065, 2281), (0.8623, 1289), (0.8951, 739)],
        30: [(0.7656, 4078), (0.8477, 2335), (0.8984, 1328), (0.9282, 754)],
        50: [(0.7774, 4143), (0.8592, 2353), (0.9135, 1334), (0.9428, 756)],
    },
    30: {  # 30 buckets
        10: [(0.6328, 4720), (0.7091, 2701), (0.7665, 1526), (0.8050, 873)],
        20: [(0.6749, 4771), (0.7530, 2717), (0.8099, 1530), (0.8487, 871)],
        30: [(0.7129, 4878), (0.7922, 2774), (0.8476, 1565), (0.8869, 888)],
        50: [(0.7288, 4919), (0.8098, 2803), (0.8655, 1571), (0.9026, 883)],
    },
    40: {  # 40 buckets
        10: [(0.6480, 5510), (0.7195, 3151), (0.7703, 1765), (0.8022, 1011)],
        20: [(0.6829, 5559), (0.7588, 3158), (0.8121, 1789), (0.8458, 1018)],
        30: [(0.7049, 5605), (0.7865, 3177), (0.8427, 1797), (0.8764, 1021)],
        50: [(0.7225, 5637), (0.8016, 3195), (0.8576, 1801), (0.8916, 1023)],
    },
    50: {  # 50 buckets
        10: [(0.6470, 6345), (0.7275, 3549), (0.7883, 1998), (0.8255, 1138)],
        20: [(0.6743, 6204), (0.7534, 3528), (0.8137, 1978), (0.8517, 1125)],
        30: [(0.6985, 6255), (0.7762, 3544), (0.8326, 1988), (0.8695, 1131)],
        50: [(0.7132, 6289), (0.7928, 3556), (0.8474, 1991), (0.8843, 1133)],
    },
    60: {  # 60 buckets
        10: [(0.6013, 6982), (0.6770, 3992), (0.7310, 2248), (0.7643, 1284)],
        20: [(0.6405, 7031), (0.7172, 3981), (0.7754, 2249), (0.8097, 1281)],
        30: [(0.6648, 7068), (0.7407, 3994), (0.7973, 2252), (0.8314, 1282)],
        50: [(0.6804, 7084), (0.7560, 3996), (0.8118, 2253), (0.8455, 1283)],
    },
}

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ===== Left Plot: Entry Points Comparison (固定 buckets=20) =====
ax1 = axes[0]
buckets_fixed = 20
colors_entry = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers_entry = ['o', 's', '^', 'D']

for i, entry_points in enumerate([10, 20, 30, 50]):
    if entry_points in data[buckets_fixed]:
        recalls = [d[0] for d in data[buckets_fixed][entry_points]]
        qps = [d[1] for d in data[buckets_fixed][entry_points]]
        ax1.plot(recalls, qps, marker=markers_entry[i], color=colors_entry[i],
                linestyle='-', linewidth=2.5, markersize=8, label=f'entry={entry_points}')

ax1.set_xlabel('Recall', fontsize=14)
ax1.set_ylabel('QPS', fontsize=14)
ax1.set_title(f'SeRF: Effect of Entry Points ({buckets_fixed} buckets)', fontsize=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=12)
ax1.set_xlim(0.60, 0.96)
ax1.set_ylim(600, 4500)

# ===== Right Plot: Buckets Comparison (固定 entry_points=30) =====
ax2 = axes[1]
entry_fixed = 30
bucket_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bucket_markers = ['o', 's', '^', 'D', 'v']
bucket_labels = ['20', '30', '40', '50', '60']

for i, buckets in enumerate([20, 30, 40, 50, 60]):
    if entry_fixed in data[buckets]:
        recalls = [d[0] for d in data[buckets][entry_fixed]]
        qps = [d[1] for d in data[buckets][entry_fixed]]
        ax2.plot(recalls, qps, marker=bucket_markers[i], color=bucket_colors[i],
                linestyle='-', linewidth=2.5, markersize=8, label=f'{buckets} buckets')

ax2.set_xlabel('Recall', fontsize=14)
ax2.set_ylabel('QPS', fontsize=14)
ax2.set_title(f'SeRF: Effect of Buckets (entry_points={entry_fixed})', fontsize=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=11, title='Buckets')
ax2.set_xlim(0.60, 0.96)
ax2.set_ylim(600, 7500)

plt.tight_layout()
plt.savefig('recall_qps_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: recall_qps_comparison.png")
plt.close()

# ===== Single comprehensive plot =====
plt.figure(figsize=(10, 8))

# Plot only representative configurations
# Use entry_points=30 as reference, show all bucket configurations
for i, buckets in enumerate([20, 30, 40, 50, 60]):
    entry_fixed = 30
    if entry_fixed in data[buckets]:
        recalls = [d[0] for d in data[buckets][entry_fixed]]
        qps = [d[1] for d in data[buckets][entry_fixed]]
        plt.plot(recalls, qps, marker=bucket_markers[i], color=bucket_colors[i],
                linestyle='-', linewidth=2.5, markersize=9, label=f'{buckets} buckets')

plt.xlabel('Recall', fontsize=16)
plt.ylabel('QPS', fontsize=16)
plt.title('SeRF: Recall vs QPS (entry_points=30, 1M Deep Dataset)', fontsize=18)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right', fontsize=13)
plt.xlim(0.60, 0.96)
plt.ylim(600, 7500)

plt.tight_layout()
plt.savefig('recall_qps_single.png', dpi=300, bbox_inches='tight')
print("Saved: recall_qps_single.png")
plt.close()

# ===== Trade-off plot: Best QPS vs Best Recall =====
fig, ax = plt.subplots(figsize=(10, 7))

# Show speed-focused (buckets=60) vs accuracy-focused (buckets=20)
configs_to_plot = [
    (20, 30, '20 buckets (accuracy-focused)', '#d62728', 'o'),
    (40, 30, '40 buckets (balanced)', '#ff7f0e', 's'),
    (60, 30, '60 buckets (speed-focused)', '#1f77b4', '^'),
]

for buckets, entry_pts, label, color, marker in configs_to_plot:
    if entry_pts in data[buckets]:
        recalls = [d[0] for d in data[buckets][entry_pts]]
        qps = [d[1] for d in data[buckets][entry_pts]]
        ax.plot(recalls, qps, marker=marker, color=color,
               linestyle='-', linewidth=3, markersize=10, label=label)

ax.set_xlabel('Recall', fontsize=16)
ax.set_ylabel('QPS', fontsize=16)
ax.set_title('SeRF: Speed-Accuracy Trade-off (entry_points=30)', fontsize=18)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=14)
ax.set_xlim(0.62, 0.90)
ax.set_ylim(800, 7200)

plt.tight_layout()
plt.savefig('recall_qps_tradeoff.png', dpi=300, bbox_inches='tight')
print("Saved: recall_qps_tradeoff.png")
plt.close()
