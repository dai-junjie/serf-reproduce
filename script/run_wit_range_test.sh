#!/bin/bash
# WIT-Image Query Range Testing Script
# Tests different query ranges: 10%, 20%, 50%, 100%
# Fixed parameters: M=64, K=100, K_Search=400

set -e

# Config
DATASET="local"
DATA_SIZE=1000000
DATASET_PATH="/home/djj/dataset/wit-image-random-1M.fvecs"
QUERY_PATH=""
QUERY_NUM=1000
QUERY_K=10

BINARY="/home/djj/code/experiment/SeRF/build/benchmark/serf_arbitrary"
OUTPUT_DIR="/home/djj/code/experiment/SeRF/results/wit_range_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COMBINED_CSV="$OUTPUT_DIR/results_${TIMESTAMP}.csv"

mkdir -p "$OUTPUT_DIR"

# Initialize combined CSV with header
echo "leap_strategy,param_type,M,K,K_Search,range_pct,recall,qps,comps" > "$COMBINED_CSV"

echo "========================================"
echo "WIT-Image Query Range Testing"
echo "========================================"
echo "Dataset: $DATASET"
echo "Data Size: $DATA_SIZE"
echo "Query Ranges: 10%, 20%, 50%, 100%"
echo "Fixed: M=64, K=100, K_Search=400"
echo "Strategies: MinLeap, MidLeap, MaxLeap"
echo "Output CSV: $COMBINED_CSV"
echo "========================================"
echo ""

# Fixed parameters
FIXED_M=64
FIXED_K=100
FIXED_KS=400

# Query ranges to test (as percentages)
RANGE_PCTS=(10 20 50 100)

# Leap strategies to test
STRATEGIES=("MIN_POS:MinLeap" "MID_POS:MidLeap" "MAX_POS:MaxLeap")

# ============================================
# Test: Vary Query Range with three leap strategies
# ============================================
echo "========================================"
echo "Test: Query Range Sweep (3 strategies)"
echo "Ranges = {10%, 20%, 50%, 100%}"
echo "M=$FIXED_M, K=$FIXED_K, K_Search=$FIXED_KS"
echo "========================================"

for STRATEGY in "${STRATEGIES[@]}"; do
    IFS=':' read -r TYPE NAME <<< "$STRATEGY"
    echo "  Running $NAME..."

    OUTPUT_FILE="$OUTPUT_DIR/temp_varyRange_${TIMESTAMP}.txt"

    $BINARY \
      -dataset "$DATASET" \
      -N $DATA_SIZE \
      -dataset_path "$DATASET_PATH" \
      -query_path "$QUERY_PATH" \
      -index_k "$FIXED_M" \
      -ef_con "$FIXED_K" \
      -ef_max "500" \
      -ef_search "$FIXED_KS" \
      -recursion_type "$TYPE" \
      > "$OUTPUT_FILE" 2>&1

    # Extract results for each range percentage
    for RANGE_PCT in "${RANGE_PCTS[@]}"; do
        RANGE_VALUE=$((DATA_SIZE * RANGE_PCT / 100))

        # Try to grep for this exact range value
        MATCH_LINE=$(grep "^range: $RANGE_VALUE[[:space:]]" "$OUTPUT_FILE" | head -1)

        if [ -n "$MATCH_LINE" ]; then
            # Parse the output: range: R  recall: REC  QPS: QPS  Comps: COMPS
            RECALL=$(echo "$MATCH_LINE" | awk '{print $4}')
            QPS=$(echo "$MATCH_LINE" | awk '{print $6}')
            COMPS=$(echo "$MATCH_LINE" | awk '{print $8}')

            echo "$NAME,Range,$FIXED_M,$FIXED_K,$FIXED_KS,$RANGE_PCT,$RECALL,$QPS,$COMPS" >> "$COMBINED_CSV"
            echo "    Range $RANGE_PCT%: recall=$RECALL, qps=$QPS"
        else
            echo "    WARNING: No data found for range $RANGE_PCT% (value=$RANGE_VALUE)"
        fi
    done

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

# Get unique range percentages
range_pcts = sorted(df['range_pct'].unique())

# Plot 1: Recall vs Range (3 strategies)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for strategy in ['MinLeap', 'MidLeap', 'MaxLeap']:
    df_s = df[df['leap_strategy'] == strategy]
    if len(df_s) > 0:
        axes[0].plot(df_s['range_pct'], df_s['recall'],
                    marker=markers[strategy], linestyle='-',
                    color=colors[strategy], linewidth=2, markersize=8,
                    label=strategy)
        axes[1].plot(df_s['range_pct'], df_s['qps'],
                    marker=markers[strategy], linestyle='-',
                    color=colors[strategy], linewidth=2, markersize=8,
                    label=strategy)

axes[0].set_xlabel('Query Range (%)')
axes[0].set_ylabel('Recall@10%')
axes[0].set_title('Recall vs Query Range')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range_pcts)
axes[0].set_xticklabels([f'{int(x)}%' for x in range_pcts])

axes[1].set_xlabel('Query Range (%)')
axes[1].set_ylabel('QPS')
axes[1].set_title('QPS vs Query Range')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range_pcts)
axes[1].set_xticklabels([f'{int(x)}%' for x in range_pcts])

plt.tight_layout()
plt.savefig(f'{output_dir}/query_range.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/query_range.png")
plt.close()

# Combined plot with log scale for QPS
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for strategy in ['MinLeap', 'MidLeap', 'MaxLeap']:
    df_s = df[df['leap_strategy'] == strategy]
    if len(df_s) > 0:
        axes[0].plot(df_s['range_pct'], df_s['recall'],
                    marker=markers[strategy], linestyle='-',
                    color=colors[strategy], linewidth=2, markersize=8,
                    label=strategy)
        axes[1].plot(df_s['range_pct'], df_s['qps'],
                    marker=markers[strategy], linestyle='-',
                    color=colors[strategy], linewidth=2, markersize=8,
                    label=strategy)

axes[0].set_xlabel('Query Range (%)')
axes[0].set_ylabel('Recall@10%')
axes[0].set_title('Recall vs Query Range')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range_pcts)
axes[0].set_xticklabels([f'{int(x)}%' for x in range_pcts])
axes[0].set_ylim([0.85, 1.0])

axes[1].set_xlabel('Query Range (%)')
axes[1].set_ylabel('QPS')
axes[1].set_title('QPS vs Query Range (Log Scale)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range_pcts)
axes[1].set_xticklabels([f'{int(x)}%' for x in range_pcts])
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(f'{output_dir}/query_range_log.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/query_range_log.png")
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
