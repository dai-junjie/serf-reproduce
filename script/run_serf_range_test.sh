#!/bin/bash
# SeRF Query Range Testing Script for Multiple Datasets
# Tests different query ranges: 10%, 20%, 50%, 100%
# Each dataset uses its own optimal parameters

set -e

BINARY="/home/djj/code/experiment/SeRF/build/benchmark/serf_arbitrary"
OUTPUT_DIR="/home/djj/code/experiment/SeRF/results/serf_range_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COMBINED_CSV="$OUTPUT_DIR/results_${TIMESTAMP}.csv"

mkdir -p "$OUTPUT_DIR"

# Initialize combined CSV with header
echo "dataset,leap_strategy,param_type,M,K,K_Search,range_pct,recall,qps,comps" > "$COMBINED_CSV"

# Query ranges to test (as percentages)
RANGE_PCTS=(10 20 50 100)

# Leap strategy - Fixed to MaxLeap only
STRATEGIES=("MAX_POS:MaxLeap")

# Dataset configurations: name:path:M:K:K_Search
# WIT-Image: M=64, K=400, K_Search=400
# SIFT: M=16, K=400, K_Search=200
# GIST: M=64, K=400, K_Search=400
declare -A DATASETS
DATASETS["WIT-2048"]="/home/djj/dataset/wit-image-random-1M.fvecs:64:400:400"
DATASETS["SIFT-128"]="/home/djj/code/experiment/SeRF/data/sift_base.fvecs:16:400:200"
DATASETS["GIST-960"]="/home/djj/code/experiment/timestampRAG/data/GIST1M/gist_base.fvecs:64:400:400"

echo "========================================"
echo "SeRF Query Range Testing - Multiple Datasets"
echo "========================================"
echo "Datasets: ${!DATASETS[@]}"
echo "Query Ranges: 10%, 20%, 50%, 100%"
echo "Strategy: MaxLeap (fixed)"
echo "Output CSV: $COMBINED_CSV"
echo ""
echo "Dataset Configurations:"
echo "  WIT-2048:   M=64,  K=400, K_Search=400"
echo "  SIFT-128:   M=16,  K=400, K_Search=200"
echo "  GIST-960:   M=64,  K=400, K_Search=400"
echo "========================================"
echo ""

# Test each dataset
for DATASET_NAME in "${!DATASETS[@]}"; do
    IFS=':' read -r DATASET_PATH FIXED_M FIXED_K FIXED_KS <<< "${DATASETS[$DATASET_NAME]}"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "WARNING: Dataset not found: $DATASET_PATH"
        continue
    fi

    DATA_SIZE=1000000

    echo "========================================"
    echo "Testing Dataset: $DATASET_NAME"
    echo "Path: $DATASET_PATH"
    echo "Config: M=$FIXED_M, K=$FIXED_K, K_Search=$FIXED_KS"
    echo "========================================"

    # Test each leap strategy
    for STRATEGY in "${STRATEGIES[@]}"; do
        IFS=':' read -r TYPE NAME <<< "$STRATEGY"
        echo "  Running $NAME..."

        OUTPUT_FILE="$OUTPUT_DIR/temp_${DATASET_NAME}_${NAME}_${TIMESTAMP}.txt"

        $BINARY \
          -dataset "local" \
          -N $DATA_SIZE \
          -dataset_path "$DATASET_PATH" \
          -query_path "" \
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

                echo "$DATASET_NAME,$NAME,Range,$FIXED_M,$FIXED_K,$FIXED_KS,$RANGE_PCT,$RECALL,$QPS,$COMPS" >> "$COMBINED_CSV"
                echo "    Range $RANGE_PCT%: recall=$RECALL, qps=$QPS"
            else
                echo "    WARNING: No data found for range $RANGE_PCT% (value=$RANGE_VALUE)"
            fi
        done

        rm "$OUTPUT_FILE"
    done
    echo "  Done with $DATASET_NAME"
    echo ""
done

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
# Plot results - Multiple datasets (MaxLeap only)
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
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Color and marker schemes for datasets
dataset_colors = {'WIT-2048': '#1f77b4', 'SIFT-128': '#ff7f0e', 'GIST-960': '#2ca02c'}
dataset_markers = {'WIT-2048': 'o', 'SIFT-128': 's', 'GIST-960': '^'}

# Get unique range percentages
range_pcts = sorted(df['range_pct'].unique())

# Combined plot - all datasets in one figure (MaxLeap only)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ds in ['WIT-2048', 'SIFT-128', 'GIST-960']:
    df_ds = df[df['dataset'] == ds]
    if len(df_ds) > 0:
        axes[0].plot(df_ds['range_pct'], df_ds['recall'],
                    marker=dataset_markers.get(ds, 'o'),
                    linestyle='-',
                    color=dataset_colors.get(ds, '#1f77b4'),
                    linewidth=2, markersize=8,
                    label=ds)
        axes[1].plot(df_ds['range_pct'], df_ds['qps'],
                    marker=dataset_markers.get(ds, 'o'),
                    linestyle='-',
                    color=dataset_colors.get(ds, '#1f77b4'),
                    linewidth=2, markersize=8,
                    label=ds)

axes[0].set_xlabel('Query Range (%)')
axes[0].set_ylabel('Recall@10')
axes[0].set_title('SeRF (MaxLeap) - Recall vs Query Range')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range_pcts)
axes[0].set_xticklabels([f'{int(x)}%' for x in range_pcts])
axes[0].set_ylim([0.85, 1.0])

axes[1].set_xlabel('Query Range (%)')
axes[1].set_ylabel('QPS')
axes[1].set_title('SeRF (MaxLeap) - QPS vs Query Range (Log Scale)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range_pcts)
axes[1].set_xticklabels([f'{int(x)}%' for x in range_pcts])
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(f'{output_dir}/serf_maxleap_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/serf_maxleap_comparison.png")
plt.close()

# Linear scale version
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ds in ['WIT-2048', 'SIFT-128', 'GIST-960']:
    df_ds = df[df['dataset'] == ds]
    if len(df_ds) > 0:
        axes[0].plot(df_ds['range_pct'], df_ds['recall'],
                    marker=dataset_markers.get(ds, 'o'),
                    linestyle='-',
                    color=dataset_colors.get(ds, '#1f77b4'),
                    linewidth=2, markersize=8,
                    label=ds)
        axes[1].plot(df_ds['range_pct'], df_ds['qps'],
                    marker=dataset_markers.get(ds, 'o'),
                    linestyle='-',
                    color=dataset_colors.get(ds, '#1f77b4'),
                    linewidth=2, markersize=8,
                    label=ds)

axes[0].set_xlabel('Query Range (%)')
axes[0].set_ylabel('Recall@10')
axes[0].set_title('SeRF (MaxLeap) - Recall vs Query Range')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range_pcts)
axes[0].set_xticklabels([f'{int(x)}%' for x in range_pcts])
axes[0].set_ylim([0.85, 1.0])

axes[1].set_xlabel('Query Range (%)')
axes[1].set_ylabel('QPS')
axes[1].set_title('SeRF (MaxLeap) - QPS vs Query Range')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range_pcts)
axes[1].set_xticklabels([f'{int(x)}%' for x in range_pcts])

plt.tight_layout()
plt.savefig(f'{output_dir}/serf_maxleap_comparison_linear.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/serf_maxleap_comparison_linear.png")
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
