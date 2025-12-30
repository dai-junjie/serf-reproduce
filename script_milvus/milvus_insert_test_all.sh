#!/bin/bash
# Milvus Insert Test - All Datasets, Different Data Sizes
# Tests IPS (Insert Per Second) based on index build time

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

# Test different data sizes (corresponding to 10%, 20%, 50%, 100% of 1M)
DATA_SIZE_LIST=(100000 200000 500000 1000000)

# Output directory
OUTPUT_DIR="./results/milvus_insert_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Milvus Insert Test - All Datasets"
echo "========================================"
echo "Datasets: ${!DATASETS[@]}"
echo "Data Sizes: ${DATA_SIZE_LIST[@]}"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Initialize CSV
CSV_FILE="$OUTPUT_DIR/insert_test_${TIMESTAMP}.csv"
echo "dataset,data_size,dim,index_type,M,efConstruction,insert_time_s,ips,build_time_s" > "$CSV_FILE"

# Test each dataset
for DATASET_NAME in "${!DATASETS[@]}"; do
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
    elif [[ "$DATASET_NAME" == *"GIST"* ]]; then
        DIM=960
        M=64
        EF_CONSTRUCTION=400
    elif [[ "$DATASET_NAME" == *"WIT"* ]]; then
        DIM=2048
        M=64
        EF_CONSTRUCTION=400
    else
        echo "ERROR: Unknown dataset $DATASET_NAME"
        continue
    fi

    echo "========================================"
    echo "Testing Dataset: $DATASET_NAME"
    echo "Path: $DATASET_PATH"
    echo "Dim: $DIM, M=$M, efConstruction=$EF_CONSTRUCTION"
    echo "========================================"

    for DATA_SIZE in "${DATA_SIZE_LIST[@]}"; do
        PERCENT=$((DATA_SIZE * 100 / 1000000))
        echo "  Size: $DATA_SIZE ($PERCENT%)"

        COLLECTION_NAME="${DATASET_NAME,,}_insert_${DATA_SIZE}"
        COLLECTION_NAME="${COLLECTION_NAME//-/_}"

        OUTPUT_FILE="$OUTPUT_DIR/temp_${DATASET_NAME}_${DATA_SIZE}.txt"

        $PYTHON "$SCRIPT_DIR/milvus_insert_test.py" \
            --host "$MILVUS_HOST" \
            --port "$MILVUS_PORT" \
            --dataset-name "$DATASET_NAME" \
            --dataset-path "$DATASET_PATH" \
            --dim "$DIM" \
            --data-size "$DATA_SIZE" \
            --collection-name "$COLLECTION_NAME" \
            --index-params "HNSW,L2,$M,$EF_CONSTRUCTION" \
            2>&1 | tee "$OUTPUT_FILE"

        # Extract results
        INSERT_TIME=$(grep "INSERT_TIME:" "$OUTPUT_FILE" | awk '{print $2}')
        BUILD_TIME=$(grep "BUILD_TIME:" "$OUTPUT_FILE" | awk '{print $2}')
        IPS=$(grep "IPS:" "$OUTPUT_FILE" | awk '{print $2}')

        if [ -n "$INSERT_TIME" ] && [ -n "$IPS" ]; then
            echo "$DATASET_NAME,$DATA_SIZE,$DIM,HNSW,$M,$EF_CONSTRUCTION,$INSERT_TIME,$IPS,$BUILD_TIME" >> "$CSV_FILE"
            echo "    IPS: $IPS, Build: ${BUILD_TIME}s"
        fi

        rm -f "$OUTPUT_FILE"
    done
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
print(df[['dataset', 'data_size', 'ips']])
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

# Plot: IPS vs Data Size
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for dataset in df['dataset'].unique():
    df_ds = df[df['dataset'] == dataset].sort_values('data_size')
    ax.plot(df_ds['data_size'], df_ds['ips'],
            marker=markers.get(dataset, 'o'),
            linestyle='-', linewidth=2.5, markersize=8,
            color=colors.get(dataset, '#1f77b4'),
            label=dataset)

ax.set_xlabel('Data Size')
ax.set_ylabel('Insert Per Second (IPS)')
ax.set_title('Milvus HNSW: Insert Performance vs Data Size')
ax.legend(title='Dataset')
ax.grid(True, alpha=0.3)
all_sizes = sorted(df['data_size'].unique())
ax.set_xticks(all_sizes)
ax.set_xticklabels([f'{int(s/1000)}K' for s in all_sizes])
plt.tight_layout()
plt.savefig(f'{output_dir}/ips_vs_size.png', dpi=150)
print(f"Saved: {output_dir}/ips_vs_size.png")
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
echo "  Plot: $OUTPUT_DIR/ips_vs_size.png"
echo "========================================"
