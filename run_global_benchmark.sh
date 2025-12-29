#!/bin/bash

# SeRF Global Range Benchmark Script
# Test global range search (no buckets)
# Test different entry points: 10, 20, 30, 50
# Test different ef_search: 128, 256, 512, 1024

BASE_PATH="data/HDFS.log-1M.fvecs"
QUERY_PATH="data/HDFS.log-1M.query.fvecs"
DATA_SIZE=1000000
QUERY_NUM=1000

OUTPUT_FILE="global_benchmark_results_1M.txt"

echo "Running SeRF Global Benchmark (1M data)..." > $OUTPUT_FILE
echo "================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

./build/benchmark/benchmark_serf_global \
    -dataset_path $BASE_PATH \
    -query_path $QUERY_PATH \
    -N $DATA_SIZE \
    -query_num $QUERY_NUM \
    -ef_search 128,256,512,1024 2>&1 | tee -a $OUTPUT_FILE

echo "" >> $OUTPUT_FILE
echo "Benchmark completed! Results saved to: $OUTPUT_FILE"
