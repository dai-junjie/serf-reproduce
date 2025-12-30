#!/bin/bash
# HNSW Baseline Insert Performance Test - Multi-Dataset Comparison
# Tests Insert Per Second (IPS) across different datasets and scales
# Each dataset uses same parameters as SeRF for fair comparison

set -e

# Dataset configurations: name:path:M:K
# Note: K_Search is not needed for insert test
# Same parameters as SeRF for fair comparison
declare -A DATASETS
DATASETS["WIT-2048"]="/home/djj/dataset/wit-image-random-1M.fvecs:64:400"
DATASETS["SIFT-128"]="/home/djj/code/experiment/SeRF/data/sift_base.fvecs:16:400"
DATASETS["GIST-960"]="/home/djj/code/experiment/timestampRAG/data/GIST1M/gist_base.fvecs:64:400"

# Test different data sizes (10%, 20%, 50%, 100% of 1M)
DATA_SIZE_LIST=(100000 200000 500000 1000000)

BINARY="./build/benchmark/benchmark_hnsw_arbitrary"
OUTPUT_DIR="./results/insert_test_hnsw"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "HNSW Baseline Insert Performance Test"
echo "========================================"
echo "Datasets: ${!DATASETS[@]}"
echo "Data Sizes (1M%): ${DATA_SIZE_LIST[@]}"
echo ""
echo "Dataset Configurations (same as SeRF):"
echo "  WIT-2048:   M=64,  K=400"
echo "  SIFT-128:  M=16,  K=400"
echo "  GIST-960:   M=64,  K=400"
echo "========================================"
echo ""

# Initialize CSV with header
echo "dataset,data_size,index_k,ef_construction,build_time,ips" > "$OUTPUT_DIR/insert_test_hnsw_${TIMESTAMP}.csv"

# Test each dataset
for DATASET_NAME in "${!DATASETS[@]}"; do
    IFS=':' read -r DATASET_PATH INDEX_K EF_CONSTRUCTION <<< "${DATASETS[$DATASET_NAME]}"

    if [ ! -f "$DATASET_PATH" ]; then
        echo "WARNING: Dataset not found: $DATASET_PATH"
        continue
    fi

    echo "========================================"
    echo "Testing Dataset: $DATASET_NAME"
    echo "Path: $DATASET_PATH"
    echo "Config: M=$INDEX_K, K=$EF_CONSTRUCTION"
    echo "========================================"

    for DATA_SIZE in "${DATA_SIZE_LIST[@]}"; do
        PERCENT=$((DATA_SIZE * 100 / 1000000))
        echo "  Size: $DATA_SIZE ($PERCENT%)"

        OUTPUT_FILE="$OUTPUT_DIR/temp_${DATASET_NAME}_${DATA_SIZE}.txt"

        # Run benchmark and capture output
        $BINARY \
            -dataset "local" \
            -N $DATA_SIZE \
            -dataset_path "$DATASET_PATH" \
            -query_path "" \
            -index_k "$INDEX_K" \
            -ef_con "$EF_CONSTRUCTION" \
            2>&1 | tee "$OUTPUT_FILE" > /dev/null

        # Extract build time from output (format: "# Build Index Time: 98.8173730s")
        BUILD_TIME_RAW=$(grep "Build Index Time" "$OUTPUT_FILE" | tail -1 | awk '{print $5}')
        # Remove trailing 's' to get pure number
        BUILD_TIME=$(echo "$BUILD_TIME_RAW" | sed 's/s$//')

        if [ -n "$BUILD_TIME" ] && [ "$BUILD_TIME" != "" ] && [[ "$BUILD_TIME" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            # Calculate IPS
            IPS=$(echo "scale=2; $DATA_SIZE / $BUILD_TIME" | bc)
            echo "    Build Time: ${BUILD_TIME}s, IPS: $IPS"
            echo "$DATASET_NAME,$DATA_SIZE,$INDEX_K,$EF_CONSTRUCTION,$BUILD_TIME,$IPS" >> "$OUTPUT_DIR/insert_test_hnsw_${TIMESTAMP}.csv"
        else
            echo "    WARNING: Could not extract build time (got: '$BUILD_TIME_RAW')"
        fi

        rm -f "$OUTPUT_FILE"
    done

    echo ""
done

echo "========================================"
echo "Insert Test Completed!"
echo ""
echo "CSV Results: $OUTPUT_DIR/insert_test_hnsw_${TIMESTAMP}.csv"
echo ""
echo "Summary:"
cat "$OUTPUT_DIR/insert_test_hnsw_${TIMESTAMP}.csv"
echo "========================================"
echo ""

# Generate plot
echo "Generating plots..."
PYTHON_SCRIPT=$(cat <<'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

csv_file = sys.argv[1]
output_dir = os.path.dirname(csv_file)

df = pd.read_csv(csv_file)
print("Data loaded:")
print(df)
print()

os.makedirs(output_dir, exist_ok=True)

# Define colors and markers for each dataset
datasets = df['dataset'].unique()
colors = {'SIFT-128': '#1f77b4', 'GIST-960': '#ff7f0e', 'WIT-2048': '#2ca02c'}
markers = {'SIFT-128': 'o', 'GIST-960': 's', 'WIT-2048': '^'}

# Create plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for dataset in datasets:
    df_ds = df[df['dataset'] == dataset].sort_values('data_size')
    ax.plot(df_ds['data_size'], df_ds['ips'],
            marker=markers.get(dataset, 'o'),
            linestyle='-',
            color=colors.get(dataset, '#1f77b4'),
            linewidth=2, markersize=8,
            label=dataset)

ax.set_xlabel('Data Size')
ax.set_ylabel('Insert Per Second (IPS)')
ax.set_title('HNSW Baseline Insert Performance vs Data Size')
ax.legend()
ax.grid(True, alpha=0.3)

# Set x-axis ticks to only show the tested sizes
all_sizes = sorted(df['data_size'].unique())
ax.set_xticks(all_sizes)
ax.set_xticklabels([f'{int(s/1000)}K' for s in all_sizes])

plt.tight_layout()
plt.savefig(f'{output_dir}/insert_performance_hnsw.png', dpi=150)
print(f"Saved: {output_dir}/insert_performance_hnsw.png")
plt.close()

# Log-scale version for better visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for dataset in datasets:
    df_ds = df[df['dataset'] == dataset].sort_values('data_size')
    ax.plot(df_ds['data_size'], df_ds['ips'],
            marker=markers.get(dataset, 'o'),
            linestyle='-',
            color=colors.get(dataset, '#1f77b4'),
            linewidth=2, markersize=8,
            label=dataset)

ax.set_xlabel('Data Size')
ax.set_ylabel('Insert Per Second (IPS)')
ax.set_title('HNSW Baseline Insert Performance vs Data Size (Log Scale)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

ax.set_xticks(all_sizes)
ax.set_xticklabels([f'{int(s/1000)}K' for s in all_sizes])

plt.tight_layout()
plt.savefig(f'{output_dir}/insert_performance_hnsw_log.png', dpi=150)
print(f"Saved: {output_dir}/insert_performance_hnsw_log.png")
plt.close()
EOF
)

cd "$OUTPUT_DIR"
python3 -c "$PYTHON_SCRIPT" "$(basename "insert_test_hnsw_${TIMESTAMP}.csv")"

echo ""
echo "========================================"
echo "Complete! Results and plots:"
echo "  CSV: $OUTPUT_DIR/insert_test_hnsw_${TIMESTAMP}.csv"
echo "  Plot: $OUTPUT_DIR/insert_performance_hnsw.png"
echo "  Log Plot: $OUTPUT_DIR/insert_performance_hnsw_log.png"
echo "========================================"
