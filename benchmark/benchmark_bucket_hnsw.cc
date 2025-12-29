/**
 * @file benchmark_bucket_hnsw.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Benchmark for Bucket-based HNSW Range Search (baseline: search first, then filter)
 * @date 2023-12-22
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "utils.h"

#include "../../src/baselines/knn_first_hnsw.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;


int main(int argc, char **argv) {
#ifdef USE_SSE
    cout << "Use SSE" << endl;
#endif
#ifdef USE_AVX
    cout << "Use AVX" << endl;
#endif
#ifdef USE_AVX512
    cout << "Use AVX512" << endl;
#endif
#ifndef NO_PARALLEL_BUILD
    cout << "Index Construct Parallelly" << endl;
#endif

    // Parameters
    string dataset = "deep";
    int data_size = 100000;
    string dataset_path = "../data/HDFS.log-1M.fvecs";
    string query_path = "../data/HDFS.log-1M.query.fvecs";

    int query_num = 10000;
    int query_k = 10;
    int num_buckets = 10;  // 分成多少个区间（buckets）
    int queries_per_bucket = 1000;  // 每个bucket多少个查询

    vector<int> index_k_list = {8};
    vector<int> ef_construction_list = {100};
    vector<int> searchef_para_list = {16, 64, 256};

    // Parse command line arguments
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") dataset = string(argv[i + 1]);
        if (arg == "-N") data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
        if (arg == "-query_path") query_path = string(argv[i + 1]);
        if (arg == "-query_num") query_num = atoi(argv[i + 1]);
        if (arg == "-b") num_buckets = atoi(argv[i + 1]);  // bucket数量
        if (arg == "-q_per_bucket") queries_per_bucket = atoi(argv[i + 1]);
        if (arg == "-index_k") index_k_list = str2vec(string(argv[i + 1]));
        if (arg == "-ef_con") ef_construction_list = str2vec(string(argv[i + 1]));
        if (arg == "-ef_search") searchef_para_list = str2vec(string(argv[i + 1]));
    }

    cout << "=== HNSW Baseline Benchmark Configuration ===" << endl;
    cout << "Dataset: " << dataset << endl;
    cout << "Data size: " << data_size << endl;
    cout << "Number of buckets: " << num_buckets << endl;
    cout << "Queries per bucket: " << queries_per_bucket << endl;
    cout << "Total queries: " << (num_buckets * queries_per_bucket) << endl;
    cout << "Query K: " << query_k << endl;
    cout << "===========================================" << endl;

    // 汇总结果: key=ef_search, value=(recall_sum, qps_sum, comps_sum, count)
    std::map<int, std::tuple<float, float, float, int>> summary_results;

    // 读取数据（只需一次）
    DataWrapper data_wrapper(num_buckets * queries_per_bucket, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);
    data_wrapper.version = "Benchmark";
    // 生成不相交的分区查询（只需一次）
    data_wrapper.generatePartitionedQueriesAndGroundtruth(num_buckets, queries_per_bucket, 0);

    for (unsigned index_k : index_k_list) {
        for (unsigned ef_construction : ef_construction_list) {

            BaseIndex::IndexParams i_params(index_k, ef_construction, ef_construction, 500);

            // 构建索引（只需一次）
            cout << "\n" << string(60, '-') << endl;
            cout << "--- Building HNSW Baseline Index ---" << endl;
            cout << "Parameters: ef_construction=" << ef_construction
                 << ", index_k=" << index_k << endl;

            KnnFirstWrapper hnsw_index(&data_wrapper);
            hnsw_index.buildIndex(&i_params);
            cout << "Index built, running searches..." << endl;

            // 测试不同的 ef_search 值
            for (auto search_ef : searchef_para_list) {
                BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "HNSW_Baseline", "bucket_benchmark");
                BaseIndex::SearchParams s_params;
                s_params.query_K = query_k;
                s_params.search_ef = search_ef;

                float recall_sum = 0;
                float time_sum = 0;
                float comps_sum = 0;
                int query_count = 0;

                timeval search_start, search_end;
                gettimeofday(&search_start, NULL);

                for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
                    int query_id = data_wrapper.query_ids[idx];
                    auto range = data_wrapper.query_ranges[idx];

                    auto res = hnsw_index.rangeFilteringSearchOutBound(
                        &s_params, &search_info,
                        data_wrapper.querys[query_id],
                        range);

                    float precision = countPrecision(data_wrapper.groundtruth[idx], res);
                    search_info.precision = precision;

                    recall_sum += precision;
                    time_sum += search_info.internal_search_time;
                    comps_sum += search_info.total_comparison;
                    query_count++;
                }

                gettimeofday(&search_end, NULL);

                // 保存该ef_search的汇总结果
                float avg_recall = recall_sum / query_count;
                float avg_qps = query_count / time_sum;
                float avg_comps = comps_sum / query_count;
                summary_results[search_ef] = {avg_recall, avg_qps, avg_comps, query_count};
            }
        }
    }

    // 输出汇总表格
    cout << "\n" << string(80, '=') << endl;
    cout << "=== SUMMARY TABLE (HNSW Baseline: " << num_buckets << " buckets) ===" << endl;
    cout << string(80, '=') << endl;

    cout << std::left << std::setw(12) << "ef_search"
         << std::setw(12) << "Recall"
         << std::setw(15) << "QPS"
         << std::setw(12) << "Comps" << endl;
    cout << string(80, '-') << endl;

    for (auto ef_search : searchef_para_list) {
        if (summary_results.find(ef_search) != summary_results.end()) {
            auto& result = summary_results[ef_search];
            float recall = std::get<0>(result);
            float qps = std::get<1>(result);
            float comps = std::get<2>(result);

            cout << std::fixed << std::setprecision(0)
                 << std::left << std::setw(12) << ef_search
                 << std::setprecision(4) << std::setw(12) << recall
                 << std::setprecision(0) << std::setw(15) << qps
                 << std::setprecision(1) << std::setw(12) << comps << endl;
        }
    }
    cout << string(80, '=') << endl;

    cout << "\n=== Benchmark completed ===" << endl;
    return 0;
}
