/*
 * OpenMP Relation Extraction Dispatcher
 * ---------------------------------------
 * Same architecture as ner_openmp.cpp: splits documents into N batches,
 * each OpenMP thread dispatches one batch to the Python/spaCy worker
 * (re_worker.py) via system(), running concurrently across CPU cores.
 *
 * Compile:
 *   g++ -fopenmp -O2 re_openmp.cpp -o re_openmp
 *
 * Run:
 *   ./re_openmp <file_list.txt> <entities.json> <num_threads> <output_dir>
 */

#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sstream>

std::vector<std::string> read_lines(const std::string &path) {
    std::vector<std::string> lines;
    std::ifstream infile(path);
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <file_list.txt> <entities.json> <num_threads> <output_dir>" << std::endl;
        return 1;
    }

    std::string file_list_path = argv[1];
    std::string entities_path = argv[2];
    int num_threads = std::atoi(argv[3]);
    std::string output_dir = argv[4];

    std::vector<std::string> all_files = read_lines(file_list_path);
    int total = all_files.size();

    if (total == 0) {
        std::cerr << "No files found in " << file_list_path << std::endl;
        return 1;
    }

    omp_set_num_threads(num_threads);
    double start_time = omp_get_wtime();

    int batch_size = (total + num_threads - 1) / num_threads;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_idx = tid * batch_size;
        int end_idx = std::min(start_idx + batch_size, total);

        if (start_idx < total) {
            std::ostringstream batch_list_path;
            batch_list_path << output_dir << "/batch_" << tid << "_files.txt";
            std::ofstream batch_file(batch_list_path.str());
            for (int i = start_idx; i < end_idx; i++) {
                batch_file << all_files[i] << "\n";
            }
            batch_file.close();

            std::ostringstream output_path;
            output_path << output_dir << "/batch_" << tid << "_result.json";

            std::ostringstream cmd;
            cmd << "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                << "python3 re_worker.py " << batch_list_path.str()
                << " " << entities_path << " " << output_path.str();

            #pragma omp critical
            {
                std::cout << "[Thread " << tid << "] Processing files "
                          << start_idx << "-" << (end_idx - 1)
                          << " (" << (end_idx - start_idx) << " docs)" << std::endl;
            }

            int ret = std::system(cmd.str().c_str());
            if (ret != 0) {
                #pragma omp critical
                std::cerr << "[Thread " << tid << "] Worker failed!" << std::endl;
            }
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "\nOpenMP Relation Extraction complete." << std::endl;
    std::cout << "Threads used: " << num_threads << std::endl;
    std::cout << "Total time: " << (end_time - start_time) << "s" << std::endl;

    return 0;
}
