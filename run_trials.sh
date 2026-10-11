#!/usr/bin/env bash
# Repeats the headline benchmarks several times and logs every timing to trial_results.csv.
#
# Run from ~/kg-parallel-project, with Neo4j already running:
#     chmod +x run_trials.sh
#     ./run_trials.sh 3
#
# The argument is the number of trials (default 3). One trial takes roughly 55 minutes.
# Re-running the script appends to trial_results.csv, so you can stop it and resume later.

set -u
cd "$(dirname "$0")"
source venv/bin/activate

TRIALS=${1:-3}
CSV=trial_results.csv
LOGDIR=trial_logs
PAUSE=30   # seconds of rest between runs so the CPU and GPU cool down

mkdir -p "$LOGDIR" ner_trial_out re_trial_out
[ -f "$CSV" ] || echo "trial,test,seconds,timestamp" > "$CSV"

# ---- sanity checks -------------------------------------------------------
missing=0
for f in data/entities.json data/relations.json data/resolved_entities.json \
         data/entities_10.json data/relations_10.json \
         openmp_output/file_list.txt ner_openmp re_openmp ner_worker.py re_worker.py \
         entity_resolution_scale.py ner_sequential.py \
         graph_population_mpi.py graph_population_mpi_scale.py; do
  if [ ! -e "$f" ]; then echo "MISSING: $f"; missing=1; fi
done
[ "$missing" -eq 1 ] && { echo "Fix the missing files above, then rerun."; exit 1; }

if ! curl -s http://localhost:7474 > /dev/null; then
  echo "Neo4j is not responding on port 7474. Run: sudo neo4j start   (wait 20 seconds)"
  exit 1
fi

# ---- helpers -------------------------------------------------------------
record() {   # trial test seconds
  echo "$1,$2,$3,$(date +%FT%T)" >> "$CSV"
  echo "[trial $1] $2 = $3 s"
}

run_timed() {   # trial test regex command...
  local trial=$1 test=$2 regex=$3
  shift 3
  local log="$LOGDIR/t${trial}_${test}.log"
  "$@" > "$log" 2>&1
  local secs
  secs=$(grep -oP "$regex" "$log" | tail -1)
  if [ -z "$secs" ]; then
    echo "[trial $trial] $test: could not read a time from $log"
    secs=NA
  fi
  record "$trial" "$test" "$secs"
  sleep "$PAUSE"
}

# ---- the trials ----------------------------------------------------------
START=${2:-1}
for t in $(seq "$START" $((START+TRIALS-1))); do
  echo "================ TRIAL $t of $TRIALS ================"

  # Stage 3: entity resolution, same batched algorithm on CPU and on GPU
  run_timed "$t" er_cpu  'Total: \K[0-9.]+' python entity_resolution_scale.py data/entities.json cpu
  run_timed "$t" er_cuda 'Total: \K[0-9.]+' python entity_resolution_scale.py data/entities.json cuda

  # Stage 1: NER, the plain script and the OpenMP dispatcher at 1 and 8 threads
  run_timed "$t" ner_seq 'Sequential NER done\. Total time: \K[0-9.]+' python ner_sequential.py
  run_timed "$t" ner_t1  'Total time: \K[0-9.]+' ./ner_openmp openmp_output/file_list.txt 1 ner_trial_out
  run_timed "$t" ner_t8  'Total time: \K[0-9.]+' ./ner_openmp openmp_output/file_list.txt 8 ner_trial_out

  # Stage 2: relation extraction with the OpenMP dispatcher at 1 and 8 threads
  run_timed "$t" re_t1 'Total time: \K[0-9.]+' ./re_openmp openmp_output/file_list.txt data/entities.json 1 re_trial_out
  run_timed "$t" re_t8 'Total time: \K[0-9.]+' ./re_openmp openmp_output/file_list.txt data/entities.json 8 re_trial_out

  # Stage 4: graph population with MPI. The warm-up run is discarded; it keeps
  # Neo4j's cold-start cost out of the 1-process measurement.
  echo "[trial $t] Neo4j warm-up (not recorded)"
  HYDRA_LAUNCHER=fork mpirun -n 4 python graph_population_mpi_scale.py \
      data/entities_10.json data/relations_10.json > "$LOGDIR/t${t}_mpi_warmup.log" 2>&1
  sleep "$PAUSE"
  export HYDRA_LAUNCHER=fork
  run_timed "$t" mpi_p1 'Total wall-clock time: \K[0-9.]+' mpirun -n 1 python graph_population_mpi.py
  run_timed "$t" mpi_p8 'Total wall-clock time: \K[0-9.]+' mpirun -n 8 python graph_population_mpi.py
done

echo
echo "All trials finished. Now run:  python summarize_trials.py"
