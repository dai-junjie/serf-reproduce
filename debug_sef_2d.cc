/**
 * @file debug_sef_2d.cc
 * @author Debug SeRF 2D Performance
 * @brief Debug why SeRF 2D has 0 recall
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "segment_graph_2d.h"
#include "utils.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char **argv) {
    cout << "=== Debug SeRF 2D Performance ===" << endl;

    // Parameters
    string dataset = "local";
    int data_size = 100000;  // Start with smaller dataset for debugging
    string dataset_path = "data/HDFS.log-1M.fvecs";
    string query_path = "data/HDFS.log-1M.query.fvecs";

    int query_k = 10;
    int test_range_size = 10000;  // Test with smaller range
    int num_tests = 10;

    // Parse command line arguments
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-N") data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
        if (arg == "-query_path") query_path = string(argv[i + 1]);
        if (arg == "-test_range") test_range_size = atoi(argv[i + 1]);
    }

    cout << "Data size: " << data_size << endl;
    cout << "Test range size: " << test_range_size << endl;
    cout << "Number of tests: " << num_tests << endl;

    // Load data
    DataWrapper data_wrapper(num_tests, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);

    cout << "Data loaded successfully" << endl;
    cout << "Data dimension: " << data_wrapper.data_dim << endl;
    cout << "Number of queries available: " << data_wrapper.querys.size() << endl;

    // Test a specific range: [data_size - test_range_size, data_size - 1]
    int l_bound = data_size - test_range_size;
    int r_bound = data_size - 1;

    cout << "\nTesting range: [" << l_bound << ", " << r_bound << "]" << endl;

    // Generate groundtruth using brute force
    cout << "Generating groundtruth..." << endl;
    vector<vector<int>> groundtruths;
    vector<int> test_query_ids;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> query_dist(0, data_wrapper.querys.size() - 1);

    for (int test = 0; test < num_tests; test++) {
        int query_id = query_dist(generator);
        test_query_ids.push_back(query_id);

        timeval t1, t2;
        gettimeofday(&t1, NULL);
        auto gt = greedyNearest(data_wrapper.nodes,
                              data_wrapper.querys[query_id],
                              l_bound, r_bound,
                              query_k);
        gettimeofday(&t2, NULL);

        groundtruths.push_back(gt);

        cout << "Test " << test << ": Query " << query_id
             << ", Groundtruth size: " << gt.size() << endl;
        if (!gt.empty()) {
            cout << "  First groundtruth ID: " << gt[0] << endl;
        }
    }

    // Initialize SeRF 2D
    cout << "\nBuilding SeRF 2D index..." << endl;
    base_hnsw::L2Space ss(data_wrapper.data_dim);

    BaseIndex::IndexParams i_params;
    i_params.ef_large_for_pruning = 0;
    i_params.ef_max = 0;
    i_params.ef_construction = 200;  // Higher ef for better recall
    i_params.K = 16;                 // Higher K for better connectivity
    i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;

    SeRF::IndexSegmentGraph2D index(&ss, &data_wrapper);
    BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D_Debug", "debug");

    timeval build_start, build_end;
    gettimeofday(&build_start, NULL);
    index.buildIndex(&i_params);
    gettimeofday(&build_end, NULL);
    float build_time = CountTime(build_start, build_end);

    cout << "SeRF 2D built in " << build_time << "s" << endl;
    cout << "Total # of neighbors: " << index.index_info->nodes_amount << endl;

    // Test SeRF 2D search with different ef values
    vector<int> ef_values = {32, 64, 128, 256, 512};

    for (int search_ef : ef_values) {
        cout << "\n=== Testing with search_ef = " << search_ef << " ===" << endl;

        BaseIndex::SearchParams s_params;
        s_params.query_K = query_k;
        s_params.search_ef = search_ef;
        s_params.query_range = r_bound - l_bound + 1;

        float total_recall = 0;
        float total_qps = 0;
        int successful_searches = 0;

        for (int test = 0; test < num_tests; test++) {
            int query_id = test_query_ids[test];
            auto range = std::make_pair(l_bound, r_bound);

            cout << "\nTest " << test << ":" << endl;
            cout << "  Query ID: " << query_id << endl;
            cout << "  Range: [" << range.first << ", " << range.second << "]" << endl;

            timeval search_start, search_end;
            gettimeofday(&search_start, NULL);

            auto res = index.rangeFilteringSearchOutBound(
                &s_params, &search_info,
                data_wrapper.querys[query_id],
                range);

            gettimeofday(&search_end, NULL);
            float search_time = CountTime(search_start, search_end);

            float precision = countPrecision(groundtruths[test], res);
            search_info.precision = precision;

            cout << "  SeRF results size: " << res.size() << endl;
            cout << "  Groundtruth size: " << groundtruths[test].size() << endl;
            cout << "  Recall: " << precision << endl;
            cout << "  Search time: " << search_time * 1000 << "ms" << endl;
            cout << "  Comparisons: " << search_info.total_comparison << endl;

            if (search_time > 0) {
                total_qps += 1.0 / search_time;
                successful_searches++;
            }
            total_recall += precision;

            // Show some result details if recall is 0
            if (precision == 0 && !groundtruths[test].empty()) {
                cout << "  DEBUG - Groundtruth IDs: ";
                for (int i = 0; i < std::min(3, (int)groundtruths[test].size()); i++) {
                    cout << groundtruths[test][i] << " ";
                }
                cout << endl;
                cout << "  DEBUG - SeRF result IDs: ";
                for (int i = 0; i < std::min(3, (int)res.size()); i++) {
                    cout << res[i] << " ";
                }
                cout << endl;
            }
        }

        if (successful_searches > 0) {
            float avg_recall = total_recall / num_tests;
            float avg_qps = total_qps / successful_searches;

            cout << "\n=== Summary for ef=" << search_ef << " ===" << endl;
            cout << "Average Recall: " << avg_recall << endl;
            cout << "Average QPS: " << avg_qps << endl;
            cout << "Successful searches: " << successful_searches << "/" << num_tests << endl;
        }
    }

    return 0;
}