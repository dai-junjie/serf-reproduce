#!/usr/bin/env python3
"""
HNSW Recall vs QPS Curve Generator (Pareto Frontier Edition)
Description:
    此脚本读取 HNSW 实验数据，绘制 Recall-QPS 权衡图。
    它会自动计算 Pareto Frontier（帕累托前沿），即在特定 Recall 下能达到的最高 QPS。
    支持单独绘制HNSW曲线，以及HNSW与SeRF的对比。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 禁用 Python 输出缓冲
sys.stdout.reconfigure(line_buffering=True)

# ============== 1. 配置区域 (CONFIGURATION) ==============

# 输入与输出路径
HNSW_INPUT_FILE = "/home/djj/code/experiment/SeRF/results/autodl/hnsw_EPYC.csv"
SERF_INPUT_FILE = "/home/djj/code/experiment/SeRF/results/autodl/results_20260102_151514.csv"
OUTPUT_DIR = "/home/djj/code/experiment/SeRF/results/autodl/plots_pareto"

# 要绘制的 Range 列表
RANGE_PCTS = [1, 10, 20, 50, 100]

# 数据集列表（需要HNSW和SeRF都有的数据集）
DATASETS = ["GIST-960", "WIT-2048"]

# ============== 2. 核心算法 (CORE FUNCTIONS) ==============

def load_data(filepath, has_header=True):
    """加载 CSV 数据"""
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        return None

    try:
        # CSV列名: dataset,method,param_type,M,K,K_Search,range_pct,recall,qps,comps,build_time,ips
        if not has_header:
            df = pd.read_csv(filepath, header=None, names=[
                'dataset', 'method', 'param_type', 'M', 'K', 'K_Search',
                'range_pct', 'recall', 'qps', 'comps', 'build_time', 'ips'
            ])
        else:
            df = pd.read_csv(filepath)

        df['dataset'] = df['dataset'].astype(str).str.strip()
        df['range_pct'] = df['range_pct'].astype(float)
        print(f"[Loaded] {os.path.basename(filepath)}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[Error] Failed to read {filepath}: {e}")
        return None

def get_pareto_frontier(df):
    """
    计算帕累托前沿 (Pareto Frontier)
    输入: 包含 'recall' 和 'qps' 列的 DataFrame
    输出: 排序后的 (recall, qps) numpy 数组，仅包含最优前沿点
    """
    if df.empty:
        return np.array([])

    # 提取点并转换为列表 [recall, qps]
    points = df[["recall", "qps"]].values.tolist()

    # 核心排序逻辑：先按 Recall 从大到小排，再按 QPS 从大到小排
    points.sort(key=lambda x: (-x[0], -x[1]))

    pareto_points = []
    current_max_qps = -1.0

    # 扫描并筛选
    for recall, qps in points:
        if qps > current_max_qps:
            pareto_points.append([recall, qps])
            current_max_qps = qps

    # 转换为 numpy 数组并按 Recall 从小到大排序
    pareto_points = np.array(pareto_points)
    if len(pareto_points) > 0:
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

    return pareto_points

def plot_hnsw_curves(df, dataset, output_dir):
    """绘制单个数据集的 HNSW 帕累托前沿曲线"""

    plt.style.use('default')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'HNSW Evaluation: {dataset}\nRecall vs QPS (Pareto Frontier)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 遍历 5 个 range
    for idx, range_pct in enumerate(RANGE_PCTS):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # 筛选当前子图的数据
        subset = df[(df["dataset"] == dataset) &
                    (df["range_pct"] == range_pct) &
                    (df["param_type"] == "All")]

        if len(subset) == 0:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", color='gray')
            ax.set_title(f"Range {range_pct}% (Empty)")
            ax.set_axis_off()
            continue

        # 绘制所有实验点（默认不显示，如需显示请取消下面注释）
        # ax.scatter(subset["recall"], subset["qps"],
        #           alpha=0.3, color='#999999', s=15, zorder=1)

        # 计算并绘制帕累托前沿
        frontier = get_pareto_frontier(subset)

        if len(frontier) > 0:
            ax.plot(frontier[:, 0], frontier[:, 1],
                   linestyle='-', linewidth=2.5, color='#d62728', alpha=0.9, zorder=2)
            ax.scatter(frontier[:, 0], frontier[:, 1],
                      marker='o', s=60, color='#d62728', edgecolors='white', linewidths=1.5, zorder=3)

            # 自动调整坐标轴范围
            min_r, max_r = subset["recall"].min(), subset["recall"].max()
            min_q, max_q = subset["qps"].min(), subset["qps"].max()

            x_margin = (max_r - min_r) * 0.05 if max_r > min_r else 0.01
            ax.set_xlim(max(0, min_r - x_margin), min(1.02, max_r + x_margin))

            y_margin = (max_q - min_q) * 0.1
            ax.set_ylim(max(0, min_q - y_margin), max_q + y_margin)

        ax.set_title(f"Range: {range_pct}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("QPS (Queries/sec)", fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

        if idx == 0:
            ax.legend(loc='lower left', framealpha=0.95, edgecolor='gray', fontsize=9)

    # 隐藏最后一个子图
    if len(RANGE_PCTS) < 6:
        axes[1, 2].set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图片
    filename = f"hnsw_{dataset.replace('-', '_')}_pareto.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[Success] Saved plot to: {filepath}")
    plt.close()

def plot_comparison(hnsw_df, serf_df, dataset, output_dir):
    """绘制 HNSW vs SeRF 对比图"""

    plt.style.use('default')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'HNSW vs SeRF Comparison: {dataset}\nRecall vs QPS (Pareto Frontier)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, range_pct in enumerate(RANGE_PCTS):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # 筛选 HNSW 数据
        hnsw_subset = hnsw_df[(hnsw_df["dataset"] == dataset) &
                              (hnsw_df["range_pct"] == range_pct) &
                              (hnsw_df["param_type"] == "All")]

        # 筛选 SeRF 数据
        serf_subset = serf_df[(serf_df["dataset"] == dataset) &
                              (serf_df["range_pct"] == range_pct) &
                              (serf_df["param_type"] == "All")]

        has_data = False

        # 绘制 HNSW
        if len(hnsw_subset) > 0:
            has_data = True
            # ax.scatter(hnsw_subset["recall"], hnsw_subset["qps"],
            #           alpha=0.2, color='#d62728', s=15, zorder=1, label='HNSW Trials')

            hnsw_frontier = get_pareto_frontier(hnsw_subset)
            if len(hnsw_frontier) > 0:
                ax.plot(hnsw_frontier[:, 0], hnsw_frontier[:, 1],
                       linestyle='-', linewidth=2.5, color='#d62728', alpha=0.9,
                       label='HNSW Pareto', zorder=2)
                ax.scatter(hnsw_frontier[:, 0], hnsw_frontier[:, 1],
                          marker='o', s=50, color='#d62728',
                          edgecolors='white', linewidths=1, zorder=3)

        # 绘制 SeRF
        if len(serf_subset) > 0:
            has_data = True
            # ax.scatter(serf_subset["recall"], serf_subset["qps"],
            #           alpha=0.2, color='#1f77b4', s=15, zorder=1, label='SeRF Trials')

            serf_frontier = get_pareto_frontier(serf_subset)
            if len(serf_frontier) > 0:
                ax.plot(serf_frontier[:, 0], serf_frontier[:, 1],
                       linestyle='-', linewidth=2.5, color='#1f77b4', alpha=0.9,
                       label='SeRF Pareto', zorder=2)
                ax.scatter(serf_frontier[:, 0], serf_frontier[:, 1],
                          marker='o', s=50, color='#1f77b4',
                          edgecolors='white', linewidths=1, zorder=3)

        if not has_data:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", color='gray')
            ax.set_title(f"Range {range_pct}% (Empty)")
            ax.set_axis_off()
            continue

        ax.set_title(f"Range: {range_pct}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("QPS (Queries/sec)", fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

        if idx == 0:
            ax.legend(loc='lower left', framealpha=0.95, edgecolor='gray', fontsize=9)

    # 隐藏最后一个子图
    if len(RANGE_PCTS) < 6:
        axes[1, 2].set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图片
    filename = f"comparison_{dataset.replace('-', '_')}_pareto.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[Success] Saved comparison plot to: {filepath}")
    plt.close()

def plot_combined_comparison(hnsw_df, serf_df, output_dir):
    """绘制所有数据集的 HNSW vs SeRF 对比图（单页）"""

    plt.style.use('default')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'HNSW vs SeRF: All Datasets\nRecall vs QPS (Pareto Frontier)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 使用固定range=10%来对比不同数据集
    FIXED_RANGE = 10

    datasets_to_plot = ["GIST-960", "WIT-960"]
    dataset_positions = [(0, 0), (0, 1)]

    # 添加不同range的GIST-960对比
    ranges_to_plot = [1, 10, 20, 50, 100]
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    for idx, (range_pct, pos) in enumerate(zip(ranges_to_plot, positions)):
        row, col = pos
        ax = axes[row, col]

        # 筛选 GIST-960 数据
        hnsw_subset = hnsw_df[(hnsw_df["dataset"] == "GIST-960") &
                              (hnsw_df["range_pct"] == range_pct) &
                              (hnsw_df["param_type"] == "All")]

        serf_subset = serf_df[(serf_df["dataset"] == "GIST-960") &
                              (serf_df["range_pct"] == range_pct) &
                              (serf_df["param_type"] == "All")]

        # 绘制 HNSW
        if len(hnsw_subset) > 0:
            # ax.scatter(hnsw_subset["recall"], hnsw_subset["qps"],
            #           alpha=0.2, color='#d62728', s=15, zorder=1)

            hnsw_frontier = get_pareto_frontier(hnsw_subset)
            if len(hnsw_frontier) > 0:
                ax.plot(hnsw_frontier[:, 0], hnsw_frontier[:, 1],
                       linestyle='-', linewidth=2.5, color='#d62728', alpha=0.9,
                       label='HNSW', zorder=2)
                ax.scatter(hnsw_frontier[:, 0], hnsw_frontier[:, 1],
                          marker='o', s=50, color='#d62728',
                          edgecolors='white', linewidths=1, zorder=3)

        # 绘制 SeRF
        if len(serf_subset) > 0:
            # ax.scatter(serf_subset["recall"], serf_subset["qps"],
            #           alpha=0.2, color='#1f77b4', s=15, zorder=1)

            serf_frontier = get_pareto_frontier(serf_subset)
            if len(serf_frontier) > 0:
                ax.plot(serf_frontier[:, 0], serf_frontier[:, 1],
                       linestyle='-', linewidth=2.5, color='#1f77b4', alpha=0.9,
                       label='SeRF', zorder=2)
                ax.scatter(serf_frontier[:, 0], serf_frontier[:, 1],
                          marker='o', s=50, color='#1f77b4',
                          edgecolors='white', linewidths=1, zorder=3)

        ax.set_title(f"GIST-960 Range: {range_pct}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("QPS (Queries/sec)", fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

        if idx == 0:
            ax.legend(loc='lower left', framealpha=0.95, edgecolor='gray', fontsize=9)

    # 隐藏最后一个子图
    axes[1, 2].set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图片
    filepath = os.path.join(output_dir, "comparison_all_gist_ranges.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[Success] Saved combined comparison plot to: {filepath}")
    plt.close()

# ============== 3. 主程序 (MAIN) ==============

def main():
    print("="*60)
    print("HNSW Plotting Tool - Pareto Frontier Edition")
    print("="*60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print("Loading Data...")
    hnsw_df = load_data(HNSW_INPUT_FILE, has_header=False)
    serf_df = load_data(SERF_INPUT_FILE, has_header=True)

    if hnsw_df is None:
        print("[Error] No HNSW data loaded. Exiting.")
        return

    print(f"HNSW datasets found: {hnsw_df['dataset'].unique().tolist()}")
    if serf_df is not None:
        print(f"SeRF datasets found: {serf_df['dataset'].unique().tolist()}")
    print("-" * 60)

    # 2. 绘制 HNSW vs SeRF 对比图
    if serf_df is not None:
        print("Generating HNSW vs SeRF comparison plots...")
        for dataset in DATASETS:
            if dataset in hnsw_df['dataset'].values and dataset in serf_df['dataset'].values:
                print(f"Processing comparison for dataset: {dataset}...")
                plot_comparison(hnsw_df, serf_df, dataset, OUTPUT_DIR)

        # 绘制综合对比图
        print("Generating combined comparison plot...")
        plot_combined_comparison(hnsw_df, serf_df, OUTPUT_DIR)
    else:
        print("[Warning] No SeRF data loaded, skipping comparison plots.")

    print("="*60)
    print("All tasks completed.")
    print(f"Check results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
