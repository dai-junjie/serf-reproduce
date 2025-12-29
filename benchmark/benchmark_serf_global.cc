/**
 * @file benchmark_serf_global.cc
 * @brief Benchmark for SeRF Global Range Search (no buckets)
 * @date 2023-12-24
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
#include "segment_graph_2d.h"
#include "utils.h"

#include <omp.h>

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
    int data_size = 1000000;
    string dataset_path = "../data/HDFS.log-1M.fvecs";
    string query_path = "../data/HDFS.log-1M.query.fvecs";

    int query_num = 10000;  // Number of queries to test
    int query_k = 10;

    vector<int> index_k_list = {8};
    vector<int> ef_construction_list = {100};
    vector<int> searchef_para_list = {128, 256, 512, 1024};  // ef_search values to test
    vector<int> num_entry_points_list = {10, 20, 30, 50};   // entry_points values to test

    // Parse command line arguments
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") dataset = string(argv[i + 1]);
        if (arg == "-N") data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
        if (arg == "-query_path") query_path = string(argv[i + 1]);
        if (arg == "-query_num") query_num = atoi(argv[i + 1]);
        if (arg == "-index_k") index_k_list = str2vec(string(argv[i + 1]));
        if (arg == "-ef_con") ef_construction_list = str2vec(string(argv[i + 1]));
        if (arg == "-ef_search") searchef_para_list = str2vec(string(argv[i + 1]));
    }

    cout << "=== SeRF Global Benchmark Configuration ===" << endl;
    cout << "Dataset: " << dataset << endl;
    cout << "Data size: " << data_size << endl;
    cout << "Query mode: GLOBAL (full range [0, " << data_size - 1 << "])" << endl;
    cout << "Number of queries: " << query_num << endl;
    cout << "Query K: " << query_k << endl;
    cout << "=========================================" << endl;

    base_hnsw::L2Space ss(1024);  // dimension

    vector<int> ef_max_list = {500};

    // 汇总结果: key=(entry_points, ef_search), value=(recall_sum, qps_sum, comps_sum, count)
    std::map<std::pair<int, int>, std::tuple<float, float, float, int>> summary_results;

    // 读取数据（只需一次）
    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);
    data_wrapper.version = "Global_Benchmark";

    // 使用多线程批量生成全局范围查询的groundtruth
    cout << "\nGenerating global range queries with groundtruth..." << endl;
    cout << "Query range: [0, " << data_size - 1 << "]" << endl;
    cout << "Using " << omp_get_max_threads() << " threads" << endl;

    data_wrapper.query_ids.resize(query_num);
    data_wrapper.query_ranges.resize(query_num);

    // 设置查询ID和范围
    for (int i = 0; i < query_num; i++) {
        data_wrapper.query_ids[i] = i;
        data_wrapper.query_ranges[i] = {0, data_size - 1};  // 全局范围
    }

    // 使用多线程批量生成groundtruth
    auto batched_gt = greedyNearestBatched(data_wrapper.nodes, data_wrapper.querys,
                                           0, data_size - 1, query_k);
    data_wrapper.groundtruth = batched_gt;

    cout << "Generated " << query_num << " global range queries with groundtruth" << endl;

    for (unsigned index_k : index_k_list) {
        for (unsigned ef_max : ef_max_list) {
            for (unsigned ef_construction : ef_construction_list) {

                BaseIndex::IndexParams i_params(index_k, ef_construction, ef_construction, ef_max);
                i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;

                // 构建索引（只需一次）
                cout << "\n" << string(60, '-') << endl;
                cout << "--- Building SeRF 2D Index ---" << endl;
                cout << "Parameters: ef_construction=" << ef_construction
                     << ", ef_max=" << ef_max
                     << ", index_k=" << index_k << endl;

                SeRF::IndexSegmentGraph2D index(&ss, &data_wrapper);
                index.buildIndex(&i_params);
                cout << "Index built, running searches..." << endl;

                // 用不同的 entry_points 和 ef_search 组合进行搜索
                for (unsigned num_entry_points : num_entry_points_list) {
                    BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D_Global", "global_benchmark");

                    cout << "\n--- Testing with entry_points=" << num_entry_points << " ---" << endl;

                    for (auto search_ef : searchef_para_list) {
                        BaseIndex::SearchParams s_params;
                        s_params.query_K = query_k;
                        s_params.search_ef = search_ef;
                        s_params.num_entry_points = num_entry_points;

                        float recall_sum = 0;
                        float time_sum = 0;
                        float comps_sum = 0;
                        int query_count = 0;

                        timeval search_start, search_end;
                        gettimeofday(&search_start, NULL);

                        for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
                            int query_id = data_wrapper.query_ids[idx];
                            auto range = data_wrapper.query_ranges[idx];

                            auto res = index.rangeFilteringSearchOutBound(
                                &s_params, &search_info,
                                data_wrapper.querys[query_id],
                                range);

                            float precision = countPrecision(data_wrapper.groundtruth[query_id], res);
                            search_info.precision = precision;

                            recall_sum += precision;
                            time_sum += search_info.internal_search_time;
                            comps_sum += search_info.total_comparison;
                            query_count++;
                        }

                        gettimeofday(&search_end, NULL);

                        // 保存该(entry_points, ef_search)的汇总结果
                        float avg_recall = recall_sum / query_count;
                        float avg_qps = query_count / time_sum;
                        float avg_comps = comps_sum / query_count;
                        summary_results[{num_entry_points, search_ef}] = {avg_recall, avg_qps, avg_comps, query_count};
                    }
                }
            }
        }
    }

    // 输出汇总表格 - 按入口点数量分组
    cout << "\n" << string(80, '=') << endl;
    cout << "=== SUMMARY TABLE (Global Range Search) ===" << endl;
    cout << string(80, '=') << endl;

    for (unsigned num_entry_points : num_entry_points_list) {
        cout << "\n--- Entry Points: " << num_entry_points << " ---" << endl;
        cout << std::left << std::setw(12) << "ef_search"
             << std::setw(12) << "Recall"
             << std::setw(15) << "QPS"
             << std::setw(12) << "Comps" << endl;
        cout << string(80, '-') << endl;

        for (auto ef_search : searchef_para_list) {
            auto key = std::make_pair(num_entry_points, ef_search);
            if (summary_results.find(key) != summary_results.end()) {
                auto& result = summary_results[key];
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
        cout << string(80, '-') << endl;
    }
    cout << string(80, '=') << endl;

    cout << "\n=== Benchmark completed ===" << endl;
    return 0;
}
