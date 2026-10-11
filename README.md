# Hybrid OpenMP-CUDA-MPI pipeline for knowledge graph construction

Four-stage pipeline on 79 SEC 10-K filings: NER and relation extraction (OpenMP),
entity resolution (CUDA), graph population in Neo4j (MPI).

Hardware: AMD Ryzen 5 5600H (6 cores, 12 threads), RTX 3050 laptop GPU, WSL2 Ubuntu.

Set the Neo4j password: export NEO4J_PASSWORD=...
Run the timing trials: ./run_trials.sh 3
Summarise the results: python3 summarize_trials.py

trial_results.csv holds the raw timings. Trials 2 and 3 ran on battery power and
are excluded in the paper.
Figures: open graphs.ipynb
