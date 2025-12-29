#!/bin/bash

# SIFT1M Bucket Benchmark Test
# 1M vectors, 128 dimensions

echo "================================"
echo "SIFT1M Bucket Benchmark Test"
echo "================================"

BASE_PATH="data/SIFT1M/sift_base.fvecs"
QUERY_PATH="data/SIFT1M/sift_query.fvecs"
DATA_SIZE=1000000
DIM=128

# 小规模测试：bucket=20, 少量 queries
NUM_BUCKETS=20
QUERIES_PER_BUCKET=100

OUTPUT_FILE="sift1m_bucket_test.txt"

echo "Dataset: SIFT1M" | tee $OUTPUT_FILE
echo "Data size: $DATA_SIZE" | tee -a $OUTPUT_FILE
echo "Dimension: $DIM" | tee -a $OUTPUT_FILE
echo "Number of buckets: $NUM_BUCKETS" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# 测试 HNSW baseline
echo "================================" | tee -a $OUTPUT_FILE
echo "Testing HNSW Baseline" | tee -a $OUTPUT_FILE
echo "================================" | tee -a $OUTPUT_FILE

./build/benchmark/benchmark_bucket_hnsw \
    -dataset sift \
    -dataset_path $BASE_PATH \
    -query_path $QUERY_PATH \
    -N $DATA_SIZE \
    -b $NUM_BUCKETS \
    -q_per_bucket $QUERIES_PER_BUCKET \
    -ef_search 128,256,512 2>&1 | tee -a $OUTPUT_FILE

echo "" | tee -a $OUTPUT_FILE
echo "================================" | tee -a $OUTPUT_FILE
echo "Testing SeRF 2D" | tee -a $OUTPUT_FILE
echo "================================" | tee -a $OUTPUT_FILE

./build/benchmark/benchmark_bucket_serf \
    -dataset sift \
    -dataset_path $BASE_PATH \
    -query_path $QUERY_PATH \
    -N $DATA_SIZE \
    -b $NUM_BUCKETS \
    -q_per_bucket $QUERIES_PER_BUCKET \
    -ef_search 128,256,512 2>&1 | tee -a $OUTPUT_FILE

echo "" | tee -a $OUTPUT_FILE
echo "Test completed! Results saved to: $OUTPUT_FILE"
