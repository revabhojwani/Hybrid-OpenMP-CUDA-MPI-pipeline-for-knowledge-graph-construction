/*
 * OpenMP NER Dispatcher
 * ----------------------
 * Splits the list of company text files into N batches (N = number of
 * OpenMP threads) and processes each batch IN PARALLEL. Each thread
 * writes the list of files it owns to a temp file, then invokes the
 * Python/spaCy worker (ner_worker.py) on that batch via system().
 *
 * This is where the actual OpenMP parallel directive does real work:
 * dispatching and waiting on independent batches concurrently, each
 * running on a separate core.
 *
 * Compile:
 *   g++ -fopenmp -O2 ner_openmp.cpp -o ner_openmp
 *
 * Run:
 *   ./ner_openmp <file_list.txt> <num_threads> <output_dir>
 *
 *   <file_list.txt> = one .txt path per line (all 79 documents)
 *   <num_threads>   = number of OpenMP threads / batches to split into
 *   <output_dir>    = where each thread writes its partial JSON result
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
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <file_list.txt> <num_threads> <output_dir>" << std::endl;
        return 1;
    }

    std::string file_list_path = argv[1];
    int num_threads = std::atoi(argv[2]);
    std::string output_dir = argv[3];

    std::vector<std::string> all_files = read_lines(file_list_path);
    int total = all_files.size();

    if (total == 0) {
        std::cerr << "No files found in " << file_list_path << std::endl;
        return 1;
    }

    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();

    // Split files into num_threads contiguous batches
    int batch_size = (total + num_threads - 1) / num_threads;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_idx = tid * batch_size;
        int end_idx = std::min(start_idx + batch_size, total);

        if (start_idx < total) {
            // Write this thread's batch of file paths to a temp list
            std::ostringstream batch_list_path;
            batch_list_path << output_dir << "/batch_" << tid << "_files.txt";
            std::ofstream batch_file(batch_list_path.str());
            for (int i = start_idx; i < end_idx; i++) {
                batch_file << all_files[i] << "\n";
            }
            batch_file.close();

            std::ostringstream output_path;
            output_path << output_dir << "/batch_" << tid << "_result.json";

            // Invoke the Python/spaCy worker on this batch.
            // Cap each worker to 1 internal thread to prevent oversubscription.
            std::ostringstream cmd;
            cmd << "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                << "python3 ner_worker.py " << batch_list_path.str()
                << " " << output_path.str();

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
    std::cout << "\nOpenMP NER complete." << std::endl;
    std::cout << "Threads used: " << num_threads << std::endl;
    std::cout << "Total time: " << (end_time - start_time) << "s" << std::endl;

    return 0;
}
