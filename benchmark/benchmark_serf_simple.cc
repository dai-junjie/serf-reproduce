/**
 * @file benchmark_serf_simple.cc
 * @author SeRF Simple Benchmark
 * @brief Simple benchmark for SeRF with average performance reporting
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "segment_graph_1d.h"
#include "segment_graph_2d.h"
#include "utils.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

struct BucketConfig {
    int num_buckets;
    int queries_per_bucket;
    vector<std::pair<int, int>> bucket_ranges;

    BucketConfig(int buckets, int queries_per_bucket)
        : num_buckets(buckets), queries_per_bucket(queries_per_bucket) {}

    void calculateRanges(int data_size) {
        bucket_ranges.clear();
        int base_range = data_size / num_buckets;
        int remainder = data_size % num_buckets;

        int start = 0;
        for (int i = 0; i < num_buckets; i++) {
            int range_size = base_range + (i < remainder ? 1 : 0);
            int end = start + range_size - 1;
            bucket_ranges.emplace_back(start, end);
            start = end + 1;
        }

        cout << "Generated " << num_buckets << " buckets" << endl;
    }
};

void print_summary(const string& method, const vector<float>& recalls,
                  const vector<float>& qps, const vector<float>& comps,
                  float build_time, int data_size) {
    if (recalls.empty()) return;

    float avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
    float avg_qps = std::accumulate(qps.begin(), qps.end(), 0.0) / qps.size();
    float avg_comps = std::accumulate(comps.begin(), comps.end(), 0.0) / comps.size();

    cout << std::setiosflags(ios::fixed) << std::setprecision(4)
         << method << " Summary:" << endl
         << "  Data Size: " << data_size << endl
         << "  Recall: " << avg_recall << endl
         << "  QPS: " << std::setprecision(0) << avg_qps << endl
         << "  Comparisons: " << std::setprecision(1) << avg_comps << endl
         << "  Build Time: " << std::setprecision(3) << build_time << "s" << endl
         << "  Queries: " << recalls.size() << endl;
}

int main(int argc, char **argv) {
    // Parameters
    string dataset = "local";
    int data_size = 1000000;  // Use full 1M data
    string dataset_path = "data/HDFS.log-1M.fvecs";
    string query_path = "data/HDFS.log-1M.query.fvecs";

    int num_buckets = 20;
    int queries_per_bucket = 25;
    int query_k = 10;

    vector<int> index_k_list = {16, 24, 32};
    vector<int> ef_construction_list = {200, 400};
    vector<int> ef_max_list = {1000, 2000};
    vector<int> searchef_para_list = {128, 256, 512, 1024};

    // Parse command line arguments
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") dataset = string(argv[i + 1]);
        if (arg == "-N") data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
        if (arg == "-query_path") query_path = string(argv[i + 1]);
        if (arg == "-num_buckets") num_buckets = atoi(argv[i + 1]);
        if (arg == "-queries_per_bucket") queries_per_bucket = atoi(argv[i + 1]);
        if (arg == "-index_k") index_k_list = str2vec(string(argv[i + 1]));
        if (arg == "-ef_con") ef_construction_list = str2vec(string(argv[i + 1]));
        if (arg == "-ef_search") searchef_para_list = str2vec(string(argv[i + 1]));
    }

    cout << "=== SeRF Simple Benchmark ===" << endl;
    cout << "Dataset: " << dataset << endl;
    cout << "Data size: " << data_size << endl;
    cout << "Buckets: " << num_buckets << " x " << queries_per_bucket << " queries" << endl;
    cout << "================================" << endl;

    // Initialize bucket configuration
    BucketConfig bucket_config(num_buckets, queries_per_bucket);
    bucket_config.calculateRanges(data_size);

    // Load data
    DataWrapper data_wrapper(num_buckets * queries_per_bucket, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);

    cout << "Data loaded: " << data_wrapper.data_dim << "D, "
         << data_wrapper.nodes.size() << " vectors" << endl;

    // Generate queries using the same approach as original benchmark
    cout << "\nGenerating queries and groundtruth..." << endl;

    // Use same range sizes as original paper
    vector<int> query_range_list;
    query_range_list.push_back(data_size * 0.001);  // 0.1%
    query_range_list.push_back(data_size * 0.005);  // 0.5%
    query_range_list.push_back(data_size * 0.01);   // 1%
    query_range_list.push_back(data_size * 0.05);   // 5%
    query_range_list.push_back(data_size * 0.1);    // 10%
    query_range_list.push_back(data_size * 0.5);    // 50%
    query_range_list.push_back(data_size);          // 100%

    cout << "Range sizes: ";
    for (int r : query_range_list) cout << r << " ";
    cout << endl;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> query_dist(0, data_wrapper.querys.size() - 1);

    int total_queries = num_buckets * queries_per_bucket;
    int queries_per_range = total_queries / query_range_list.size();

    for (auto range_size : query_range_list) {
        for (int q = 0; q < queries_per_range; q++) {
            int query_idx = query_dist(generator);

            // Random starting position for the range (like original benchmark)
            int l_bound, r_bound;
            if (range_size == data_size) {
                l_bound = 0;
                r_bound = data_size - 1;
            } else {
                std::uniform_int_distribution<int> range_dist(0, data_size - range_size - 80);
                l_bound = range_dist(generator);
                r_bound = l_bound + range_size - 1;
            }

            data_wrapper.query_ids.push_back(query_idx);
            data_wrapper.query_ranges.push_back(std::make_pair(l_bound, r_bound));

            // Generate groundtruth
            auto gt = greedyNearest(data_wrapper.nodes,
                                  data_wrapper.querys[query_idx],
                                  l_bound, r_bound,
                                  query_k);
            data_wrapper.groundtruth.push_back(gt);
        }
    }

    cout << "Generated " << data_wrapper.query_ids.size() << " queries" << endl;

    // Initialize space
    base_hnsw::L2Space ss(data_wrapper.data_dim);

    cout << "\n=== Benchmarking SeRF ===" << endl;

    // Test SeRF 1D
    cout << "\n--- SeRF 1D (Half-Bounded) ---" << endl;
    for (unsigned index_k : index_k_list) {
        for (unsigned ef_construction : ef_construction_list) {
            cout << "\nTesting: index_k=" << index_k << ", ef_con=" << ef_construction << endl;

            BaseIndex::IndexParams i_params;
            i_params.ef_large_for_pruning = 0;
            i_params.ef_max = 0;
            i_params.ef_construction = ef_construction;
            i_params.K = index_k;
            i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;

            SeRF::IndexSegmentGraph1D<float> index(&ss, &data_wrapper);
            BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_1D", "simple");

            // Build index
            timeval build_start, build_end;
            gettimeofday(&build_start, NULL);
            index.buildIndex(&i_params);
            gettimeofday(&build_end, NULL);
            float build_time = CountTime(build_start, build_end);

            cout << "Index built: " << build_time << "s, "
                 << index.index_info->nodes_amount << " neighbors" << endl;

            // Test with different ef values
            for (auto search_ef : searchef_para_list) {
                BaseIndex::SearchParams s_params;
                s_params.query_K = query_k;
                s_params.search_ef = search_ef;

                vector<float> recalls, qps, comps;

                for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
                    int query_id = data_wrapper.query_ids[idx];
                    auto range = data_wrapper.query_ranges[idx];
                    s_params.query_range = range.second - range.first + 1;

                    auto res = index.rangeFilteringSearchInRange(
                        &s_params, &search_info,
                        data_wrapper.querys[query_id],
                        range);

                    float precision = countPrecision(data_wrapper.groundtruth[idx], res);
                    recalls.push_back(precision);
                    qps.push_back(1.0 / search_info.internal_search_time);
                    comps.push_back(search_info.total_comparison);
                }

                print_summary("SeRF_1D", recalls, qps, comps, build_time, data_size);
                cout << "  Parameters: ef_search=" << search_ef << endl;
            }
        }
    }

    // Test SeRF 2D
    cout << "\n--- SeRF 2D (Arbitrary Range) ---" << endl;
    for (unsigned index_k : index_k_list) {
        for (unsigned ef_max : ef_max_list) {
            for (unsigned ef_construction : ef_construction_list) {
                cout << "\nTesting: index_k=" << index_k << ", ef_max=" << ef_max << ", ef_con=" << ef_construction << endl;

                // Use the same parameter pattern as original serf_arbitrary.cc
                BaseIndex::IndexParams i_params(index_k, ef_construction, ef_construction, ef_max);
                i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;

                SeRF::IndexSegmentGraph2D index(&ss, &data_wrapper);
                BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D", "simple");

                // Build index
                timeval build_start, build_end;
                gettimeofday(&build_start, NULL);
                index.buildIndex(&i_params);
                gettimeofday(&build_end, NULL);
                float build_time = CountTime(build_start, build_end);

                cout << "Index built: " << build_time << "s, "
                     << index.index_info->nodes_amount << " neighbors" << endl;

                // Test with different ef values
                for (auto search_ef : searchef_para_list) {
                    BaseIndex::SearchParams s_params;
                    s_params.query_K = query_k;
                    s_params.search_ef = search_ef;

                    vector<float> recalls, qps, comps;

                    for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
                        int query_id = data_wrapper.query_ids[idx];
                        auto range = data_wrapper.query_ranges[idx];
                        s_params.query_range = range.second - range.first + 1;

                        auto res = index.rangeFilteringSearchOutBound(
                            &s_params, &search_info,
                            data_wrapper.querys[query_id],
                            range);

                        float precision = countPrecision(data_wrapper.groundtruth[idx], res);
                        recalls.push_back(precision);
                        qps.push_back(1.0 / search_info.internal_search_time);
                        comps.push_back(search_info.total_comparison);
                    }

                    print_summary("SeRF_2D", recalls, qps, comps, build_time, data_size);
                    cout << "  Parameters: ef_search=" << search_ef << endl;
                }
            }
        }
    }

    cout << "\n=== Benchmark completed ===" << endl;

    return 0;
}