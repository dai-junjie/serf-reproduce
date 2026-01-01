#!/bin/bash
# Milvus Search Test - All Datasets, 1M Data, Different Query Ranges
# Tests Recall and QPS vs Query Range

set -e

# Python environment
PYTHON="/home/djj/miniconda3/envs/ai/bin/python"

# Milvus connection
MILVUS_HOST="localhost"
MILVUS_PORT="19530"

# Dataset configurations
declare -A DATASETS
DATASETS["SIFT-128"]="/home/djj/code/experiment/SeRF/data/sift_base.fvecs"
DATASETS["GIST-960"]="/home/djj/code/experiment/timestampRAG/data/GIST1M/gist_base.fvecs"
DATASETS["WIT-2048"]="/home/djj/dataset/wit-image-random-1M.fvecs"

# Fixed data size for search test
DATA_SIZE=1000000

# Query ranges to test (as percentages)
RANGE_PCTS=(1 10 20 50 100)

# Search parameters
SEARCH_RUNS=10
TOP_K=10

# Output directory
OUTPUT_DIR="./results/milvus_search_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Milvus Search Test - All Datasets"
echo "========================================"
echo "Datasets: ${!DATASETS[@]}"
echo "Data Size: $DATA_SIZE (1M)"
echo "Query Ranges: ${RANGE_PCTS[@]}%"
echo "Search Runs: $SEARCH_RUNS"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Initialize CSV
CSV_FILE="$OUTPUT_DIR/search_test_${TIMESTAMP}.csv"
echo "dataset,data_size,dim,M,efConstruction,ef,range_pct,range_width,latency_ms,qps,recall,result_count" > "$CSV_FILE"

# Test each dataset in explicit order
for DATASET_NAME in "SIFT-128" "GIST-960" "WIT-2048"; do
    DATASET_PATH="${DATASETS[$DATASET_NAME]}"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "WARNING: Dataset not found: $DATASET_PATH"
        continue
    fi

    # Get dimension from dataset name
    if [[ "$DATASET_NAME" == *"SIFT"* ]]; then
        DIM=128
        M=16
        EF_CONSTRUCTION=400
        EF_SEARCH=200
    elif [[ "$DATASET_NAME" == *"GIST"* ]]; then
        DIM=960
        M=64
        EF_CONSTRUCTION=400
        EF_SEARCH=400
    elif [[ "$DATASET_NAME" == *"WIT"* ]]; then
        DIM=2048
        M=64
        EF_CONSTRUCTION=400
        EF_SEARCH=400
    else
        echo "ERROR: Unknown dataset $DATASET_NAME"
        continue
    fi

    COLLECTION_NAME="${DATASET_NAME,,}_search_1m"
    COLLECTION_NAME="${COLLECTION_NAME//-/_}"

    echo "========================================"
    echo "Testing Dataset: $DATASET_NAME"
    echo "Path: $DATASET_PATH"
    echo "Dim: $DIM, M=$M, efConstruction=$EF_CONSTRUCTION, ef=$EF_SEARCH"
    echo "========================================"

    OUTPUT_FILE="$OUTPUT_DIR/temp_${DATASET_NAME}.txt"

    $PYTHON "$SCRIPT_DIR/milvus_search_test.py" \
        --host "$MILVUS_HOST" \
        --port "$MILVUS_PORT" \
        --dataset-name "$DATASET_NAME" \
        --dataset-path "$DATASET_PATH" \
        --dim "$DIM" \
        --data-size "$DATA_SIZE" \
        --collection-name "$COLLECTION_NAME" \
        --index-params "HNSW,L2,$M,$EF_CONSTRUCTION" \
        --search-params "L2,$EF_SEARCH" \
        --range-pcts "${RANGE_PCTS[@]}" \
        --runs "$SEARCH_RUNS" \
        --top-k "$TOP_K" \
        2>&1 | tee "$OUTPUT_FILE"

    # Extract results
    for RANGE_PCT in "${RANGE_PCTS[@]}"; do
        RANGE_LINE=$(grep "RANGE_${RANGE_PCT}PCT:" "$OUTPUT_FILE" | head -1)
        if [ -n "$RANGE_LINE" ]; then
            LATENCY=$(echo "$RANGE_LINE" | awk '{print $3}')
            QPS=$(echo "$RANGE_LINE" | awk '{print $5}')
            RECALL=$(echo "$RANGE_LINE" | awk '{print $7}')
            COUNT=$(echo "$RANGE_LINE" | awk '{print $9}')
            RANGE_WIDTH=$((DATA_SIZE * RANGE_PCT / 100))
            echo "$DATASET_NAME,$DATA_SIZE,$DIM,$M,$EF_CONSTRUCTION,$EF_SEARCH,$RANGE_PCT,$RANGE_WIDTH,$LATENCY,$QPS,$RECALL,$COUNT" >> "$CSV_FILE"
            echo "  Range ${RANGE_PCT}%: QPS=$QPS, Recall=$RECALL, Latency=${LATENCY}ms"
        fi
    done

    rm -f "$OUTPUT_FILE"
    echo ""
done

echo "========================================"
echo "All tests completed!"
echo "Results: $CSV_FILE"
echo "========================================"
echo ""

# Generate plots
echo "Generating plots..."
PYTHON_SCRIPT=$(cat <<'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file = sys.argv[1]
output_dir = os.path.dirname(csv_file)

df = pd.read_csv(csv_file)
print("Data loaded:")
print(df[['dataset', 'range_pct', 'qps', 'recall']])
print()

os.makedirs(output_dir, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Colors and markers
colors = {'SIFT-128': '#1f77b4', 'GIST-960': '#ff7f0e', 'WIT-2048': '#2ca02c'}
markers = {'SIFT-128': 'o', 'GIST-960': 's', 'WIT-2048': '^'}

# Plot 1: Recall vs Query Range
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for dataset in df['dataset'].unique():
    df_ds = df[df['dataset'] == dataset].sort_values('range_pct')
    ax.plot(df_ds['range_pct'], df_ds['recall'],
            marker=markers.get(dataset, 'o'),
            linestyle='-', linewidth=2.5, markersize=8,
            color=colors.get(dataset, '#1f77b4'),
            label=dataset)

ax.set_xlabel('Query Range (%)')
ax.set_ylabel('Recall@10')
ax.set_title('Milvus HNSW: Recall vs Query Range (1M)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df['range_pct'].unique()))
ax.set_xticklabels([f'{int(x)}%' for x in sorted(df['range_pct'].unique())])
ax.set_ylim([0.85, 1.02])
plt.tight_layout()
plt.savefig(f'{output_dir}/recall_vs_range.png', dpi=150)
print(f"Saved: {output_dir}/recall_vs_range.png")
plt.close()

# Plot 2: QPS vs Query Range
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for dataset in df['dataset'].unique():
    df_ds = df[df['dataset'] == dataset].sort_values('range_pct')
    ax.plot(df_ds['range_pct'], df_ds['qps'],
            marker=markers.get(dataset, 'o'),
            linestyle='-', linewidth=2.5, markersize=8,
            color=colors.get(dataset, '#1f77b4'),
            label=dataset)

ax.set_xlabel('Query Range (%)')
ax.set_ylabel('QPS')
ax.set_title('Milvus HNSW: QPS vs Query Range (1M)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df['range_pct'].unique()))
ax.set_xticklabels([f'{int(x)}%' for x in sorted(df['range_pct'].unique())])
plt.tight_layout()
plt.savefig(f'{output_dir}/qps_vs_range.png', dpi=150)
print(f"Saved: {output_dir}/qps_vs_range.png")
plt.close()

# Plot 3: Combined (Recall and QPS)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for dataset in df['dataset'].unique():
    df_ds = df[df['dataset'] == dataset].sort_values('range_pct')
    axes[0].plot(df_ds['range_pct'], df_ds['recall'],
                marker=markers.get(dataset, 'o'),
                linestyle='-', linewidth=2.5, markersize=8,
                color=colors.get(dataset, '#1f77b4'),
                label=dataset)
    axes[1].plot(df_ds['range_pct'], df_ds['qps'],
                marker=markers.get(dataset, 'o'),
                linestyle='-', linewidth=2.5, markersize=8,
                color=colors.get(dataset, '#1f77b4'),
                label=dataset)

axes[0].set_xlabel('Query Range (%)')
axes[0].set_ylabel('Recall@10')
axes[0].set_title('Recall vs Query Range')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(sorted(df['range_pct'].unique()))
axes[0].set_xticklabels([f'{int(x)}%' for x in sorted(df['range_pct'].unique())])
axes[0].set_ylim([0.85, 1.02])

axes[1].set_xlabel('Query Range (%)')
axes[1].set_ylabel('QPS')
axes[1].set_title('QPS vs Query Range')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(sorted(df['range_pct'].unique()))
axes[1].set_xticklabels([f'{int(x)}%' for x in sorted(df['range_pct'].unique())])

plt.tight_layout()
plt.savefig(f'{output_dir}/combined_vs_range.png', dpi=150)
print(f"Saved: {output_dir}/combined_vs_range.png")
plt.close()

print("\nAll plots saved!")
EOF
)

cd "$OUTPUT_DIR"
$PYTHON -c "$PYTHON_SCRIPT" "$(basename "$CSV_FILE")"

echo ""
echo "========================================"
echo "Complete! Results and plots:"
echo "  CSV: $CSV_FILE"
echo "  Plots: $OUTPUT_DIR/"
echo "    - recall_vs_range.png"
echo "    - qps_vs_range.png"
echo "    - combined_vs_range.png"
echo "========================================"
