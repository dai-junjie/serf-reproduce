#!/bin/bash

# SeRF Bucket Benchmark Script
# Test different bucket sizes from 20 to 100
# Test different entry points: 10, 15, 20

BASE_PATH="data/HDFS.log-1M.fvecs"
QUERY_PATH="data/HDFS.log-1M.query.fvecs"
DATA_SIZE=1000000
QUERIES_PER_BUCKET=1000

OUTPUT_FILE="bucket_benchmark_results_1M.txt"

echo "Running SeRF Bucket Benchmark (1M data)..." > $OUTPUT_FILE
echo "================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for B in 20 30 40 50 60; do
    echo "================================" | tee -a $OUTPUT_FILE
    echo "Running with -b $B" | tee -a $OUTPUT_FILE
    echo "================================" | tee -a $OUTPUT_FILE

    ./build/benchmark/benchmark_bucket_serf \
        -dataset_path $BASE_PATH \
        -query_path $QUERY_PATH \
        -N $DATA_SIZE \
        -b $B \
        -q_per_bucket $QUERIES_PER_BUCKET \
        -ef_search 128,256,512,1024 2>&1 | tee -a $OUTPUT_FILE

    echo "" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
done

echo "All benchmarks completed!" | tee -a $OUTPUT_FILE
echo "Results saved to: $OUTPUT_FILE"
