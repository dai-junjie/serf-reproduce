#!/bin/bash
# WIT-Image Parameter Testing Script (Paper Experiment)
# Tests M, K, K_Search parameters with THREE Leap strategies
# Query range fixed at 10% of data size

set -e

# Config
DATASET="local"
DATA_SIZE=1000000
DATASET_PATH="/home/djj/dataset/wit-image.fvecs"
QUERY_PATH=""
QUERY_NUM=1000
QUERY_K=10

BINARY="/home/djj/code/experiment/SeRF/build/benchmark/serf_arbitrary"
OUTPUT_DIR="/home/djj/code/experiment/SeRF/results/wit_param_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COMBINED_CSV="$OUTPUT_DIR/results_${TIMESTAMP}.csv"

mkdir -p "$OUTPUT_DIR"

# Initialize combined CSV with header
echo "leap_strategy,param_type,M,K,K_Search,recall,qps,comps" > "$COMBINED_CSV"

echo "========================================"
echo "WIT-Image Parameter Testing (Paper Setup)"
echo "========================================"
echo "Dataset: $DATASET"
echo "Data Size: $DATA_SIZE"
echo "Query Range: 10% (range = $((DATA_SIZE / 10)))"
echo "Strategies: MinLeap, MidLeap, MaxLeap"
echo "Output CSV: $COMBINED_CSV"
echo "========================================"
echo ""

# Paper default values
DEFAULT_M=8
DEFAULT_K=100
DEFAULT_KS=400

# Leap strategies to test
STRATEGIES=("MIN_POS:MinLeap" "MID_POS:MidLeap" "MAX_POS:MaxLeap")

# ============================================
# Test 1: Vary M with three leap strategies
# ============================================
echo "========================================"
echo "Test 1: Vary M parameter (3 strategies)"
echo "M = {8, 16, 32, 64}, K=$DEFAULT_K, K_Search=$DEFAULT_KS"
echo "========================================"

for STRATEGY in "${STRATEGIES[@]}"; do
    IFS=':' read -r TYPE NAME <<< "$STRATEGY"
    echo "  Running $NAME..."

    OUTPUT_FILE="$OUTPUT_DIR/temp_varyM_${TIMESTAMP}.txt"

    $BINARY \
      -dataset "$DATASET" \
      -N $DATA_SIZE \
      -dataset_path "$DATASET_PATH" \
      -query_path "$QUERY_PATH" \
      -index_k "8,16,32,64" \
      -ef_con "$DEFAULT_K" \
      -ef_max "500" \
      -ef_search "$DEFAULT_KS" \
      -recursion_type "$TYPE" \
      > "$OUTPUT_FILE" 2>&1

    grep "^range: $((DATA_SIZE / 10))[[:space:]]" "$OUTPUT_FILE" | awk -v name="$NAME" '
        BEGIN { m_vals[0]=8; m_vals[1]=16; m_vals[2]=32; m_vals[3]=64; idx=0; }
        {
            printf "%s,M,%d,100,400,%s,%s,%s\n", name, m_vals[idx], $4, $6, $8;
            idx++;
        }
    ' >> "$COMBINED_CSV"

    rm "$OUTPUT_FILE"
done
echo "  Done"
echo ""

# ============================================
# Test 2: Vary K with three leap strategies
# ============================================
echo "========================================"
echo "Test 2: Vary K parameter (3 strategies)"
echo "M=$DEFAULT_M, K = {100, 200, 400, 800}, K_Search=$DEFAULT_KS"
echo "========================================"

for STRATEGY in "${STRATEGIES[@]}"; do
    IFS=':' read -r TYPE NAME <<< "$STRATEGY"
    echo "  Running $NAME..."

    OUTPUT_FILE="$OUTPUT_DIR/temp_varyK_${TIMESTAMP}.txt"

    $BINARY \
      -dataset "$DATASET" \
      -N $DATA_SIZE \
      -dataset_path "$DATASET_PATH" \
      -query_path "$QUERY_PATH" \
      -index_k "$DEFAULT_M" \
      -ef_con "100,200,400,800" \
      -ef_max "500" \
      -ef_search "$DEFAULT_KS" \
      -recursion_type "$TYPE" \
      > "$OUTPUT_FILE" 2>&1

    grep "^range: $((DATA_SIZE / 10))[[:space:]]" "$OUTPUT_FILE" | awk -v name="$NAME" '
        BEGIN { k_vals[0]=100; k_vals[1]=200; k_vals[2]=400; k_vals[3]=800; idx=0; }
        {
            printf "%s,K,64,%d,400,%s,%s,%s\n", name, k_vals[idx], $4, $6, $8;
            idx++;
        }
    ' >> "$COMBINED_CSV"

    rm "$OUTPUT_FILE"
done
echo "  Done"
echo ""

# ============================================
# Test 3: Vary K_Search with three leap strategies
# ============================================
echo "========================================"
echo "Test 3: Vary K_Search parameter (3 strategies)"
echo "M=$DEFAULT_M, K=$DEFAULT_K, K_Search = {100, 200, 300, 400}"
echo "========================================"

for STRATEGY in "${STRATEGIES[@]}"; do
    IFS=':' read -r TYPE NAME <<< "$STRATEGY"
    echo "  Running $NAME..."

    OUTPUT_FILE="$OUTPUT_DIR/temp_varyKS_${TIMESTAMP}.txt"

    $BINARY \
      -dataset "$DATASET" \
      -N $DATA_SIZE \
      -dataset_path "$DATASET_PATH" \
      -query_path "$QUERY_PATH" \
      -index_k "$DEFAULT_M" \
      -ef_con "$DEFAULT_K" \
      -ef_max "500" \
      -ef_search "100,200,300,400" \
      -recursion_type "$TYPE" \
      > "$OUTPUT_FILE" 2>&1

    grep "^range: $((DATA_SIZE / 10))[[:space:]]" "$OUTPUT_FILE" | awk -v name="$NAME" '
        BEGIN { ks_vals[0]=100; ks_vals[1]=200; ks_vals[2]=300; ks_vals[3]=400; idx=0; }
        {
            printf "%s,K_Search,64,100,%d,%s,%s,%s\n", name, ks_vals[idx], $4, $6, $8;
            idx++;
        }
    ' >> "$COMBINED_CSV"

    rm "$OUTPUT_FILE"
done
echo "  Done"
echo ""

echo "========================================"
echo "All tests completed!"
echo ""
echo "Results saved to: $COMBINED_CSV"
echo ""
echo "Sample data:"
cat "$COMBINED_CSV"
echo ""
echo "Total records: $(wc -l < "$COMBINED_CSV")"
echo "========================================"
echo ""

# ============================================
# Plot results with 3 lines per figure
# ============================================
echo "========================================"
echo "Generating plots..."
echo "========================================"

# Create Python script inline
PYTHON_SCRIPT=$(cat <<'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

csv_file = sys.argv[1]
output_dir = sys.argv[2]

df = pd.read_csv(csv_file)
print("Data loaded:")
print(df)
print()

os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Color map for strategies
colors = {'MinLeap': '#1f77b4', 'MidLeap': '#ff7f0e', 'MaxLeap': '#2ca02c'}
markers = {'MinLeap': 'o', 'MidLeap': 's', 'MaxLeap': '^'}

# Plot M parameter
df_M = df[df['param_type'] == 'M'].copy()
if len(df_M) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    m_vals = sorted(df_M['M'].unique())
    for strategy in ['MinLeap', 'MidLeap', 'MaxLeap']:
        df_s = df_M[df_M['leap_strategy'] == strategy]
        if len(df_s) > 0:
            axes[0].plot(df_s['M'], df_s['recall'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
            axes[1].plot(df_s['M'], df_s['qps'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
    axes[0].set_xlabel('M')
    axes[0].set_ylabel('Recall@10%')
    axes[0].set_title('Recall vs M')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.85, 1.0])
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(m_vals)
    axes[0].set_xticklabels(m_vals)
    axes[1].set_xlabel('M')
    axes[1].set_ylabel('QPS')
    axes[1].set_title('QPS vs M')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks(m_vals)
    axes[1].set_xticklabels(m_vals)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/M_parameter.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/M_parameter.png")
    plt.close()

# Plot K parameter
df_K = df[df['param_type'] == 'K'].copy()
if len(df_K) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    k_vals = sorted(df_K['K'].unique())
    for strategy in ['MinLeap', 'MidLeap', 'MaxLeap']:
        df_s = df_K[df_K['leap_strategy'] == strategy]
        if len(df_s) > 0:
            axes[0].plot(df_s['K'], df_s['recall'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
            axes[1].plot(df_s['K'], df_s['qps'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Recall@10%')
    axes[0].set_title('Recall vs K')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.85, 1.0])
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(k_vals)
    axes[0].set_xticklabels(k_vals)
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('QPS')
    axes[1].set_title('QPS vs K')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks(k_vals)
    axes[1].set_xticklabels(k_vals)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/K_parameter.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/K_parameter.png")
    plt.close()

# Plot K_Search parameter
df_KS = df[df['param_type'] == 'K_Search'].copy()
if len(df_KS) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ks_vals = sorted(df_KS['K_Search'].unique())
    for strategy in ['MinLeap', 'MidLeap', 'MaxLeap']:
        df_s = df_KS[df_KS['leap_strategy'] == strategy]
        if len(df_s) > 0:
            axes[0].plot(df_s['K_Search'], df_s['recall'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
            axes[1].plot(df_s['K_Search'], df_s['qps'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
    axes[0].set_xlabel('K_Search')
    axes[0].set_ylabel('Recall@10%')
    axes[0].set_title('Recall vs K_Search')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.85, 1.0])
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(ks_vals)
    axes[0].set_xticklabels(ks_vals)
    axes[1].set_xlabel('K_Search')
    axes[1].set_ylabel('QPS')
    axes[1].set_title('QPS vs K_Search')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks(ks_vals)
    axes[1].set_xticklabels(ks_vals)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/K_Search_parameter.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/K_Search_parameter.png")
    plt.close()

# Combined plot (2x3 with all 6 subplots)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# M parameter plots
if len(df_M) > 0:
    m_vals = sorted(df_M['M'].unique())
    for strategy in ['MinLeap', 'MidLeap', 'MaxLeap']:
        df_s = df_M[df_M['leap_strategy'] == strategy]
        if len(df_s) > 0:
            axes[0, 0].plot(df_s['M'], df_s['recall'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
            axes[1, 0].plot(df_s['M'], df_s['qps'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
    axes[0, 0].set_ylabel('Recall@10%')
    axes[0, 0].set_title('Recall vs M')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.85, 1.0])
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].set_xticks(m_vals)
    axes[0, 0].set_xticklabels(m_vals)
    axes[1, 0].set_xlabel('M')
    axes[1, 0].set_ylabel('QPS')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].set_xticks(m_vals)
    axes[1, 0].set_xticklabels(m_vals)

# K parameter plots
if len(df_K) > 0:
    k_vals = sorted(df_K['K'].unique())
    for strategy in ['MinLeap', 'MidLeap', 'MaxLeap']:
        df_s = df_K[df_K['leap_strategy'] == strategy]
        if len(df_s) > 0:
            axes[0, 1].plot(df_s['K'], df_s['recall'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
            axes[1, 1].plot(df_s['K'], df_s['qps'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
    axes[0, 1].set_ylabel('Recall@10%')
    axes[0, 1].set_title('Recall vs K')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.85, 1.0])
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].set_xticks(k_vals)
    axes[0, 1].set_xticklabels(k_vals)
    axes[1, 1].set_xlabel('K')
    axes[1, 1].set_ylabel('QPS')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log', base=2)
    axes[1, 1].set_xticks(k_vals)
    axes[1, 1].set_xticklabels(k_vals)

# K_Search parameter plots
if len(df_KS) > 0:
    ks_vals = sorted(df_KS['K_Search'].unique())
    for strategy in ['MinLeap', 'MidLeap', 'MaxLeap']:
        df_s = df_KS[df_KS['leap_strategy'] == strategy]
        if len(df_s) > 0:
            axes[0, 2].plot(df_s['K_Search'], df_s['recall'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
            axes[1, 2].plot(df_s['K_Search'], df_s['qps'],
                        marker=markers[strategy], linestyle='-',
                        color=colors[strategy], linewidth=2, markersize=8,
                        label=strategy)
    axes[0, 2].set_ylabel('Recall@10%')
    axes[0, 2].set_title('Recall vs K_Search')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0.85, 1.0])
    axes[0, 2].set_xscale('log', base=2)
    axes[0, 2].set_xticks(ks_vals)
    axes[0, 2].set_xticklabels(ks_vals)
    axes[1, 2].set_xlabel('K_Search')
    axes[1, 2].set_ylabel('QPS')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xscale('log', base=2)
    axes[1, 2].set_xticks(ks_vals)
    axes[1, 2].set_xticklabels(ks_vals)

plt.tight_layout()
plt.savefig(f'{output_dir}/all_parameters_combined.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/all_parameters_combined.png")
plt.close()

print("\nAll plots saved!")
EOF
)

# Run Python script
PLOTS_DIR="$OUTPUT_DIR/plots_${TIMESTAMP}"
cd "$OUTPUT_DIR"
python3 -c "$PYTHON_SCRIPT" "$(basename "$COMBINED_CSV")" "plots_${TIMESTAMP}"

echo ""
echo "========================================"
echo "Complete! Results and plots:"
echo "  CSV: $COMBINED_CSV"
echo "  Plots: $PLOTS_DIR"
echo "========================================"
