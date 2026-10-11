"""
Reads trial_results.csv (written by run_trials.sh) and prints mean, standard deviation,
speedup, parallel efficiency and worker-seconds for every stage.

Run from ~/kg-parallel-project:
    python summarize_trials.py

It also writes trial_summary.txt, which you can paste back into the chat.
"""
import csv
import statistics as st
from collections import defaultdict

CSV_PATH = "trial_results.csv"

times = defaultdict(list)
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        if row['trial'].strip() in ('2', '3'):
            continue  # trials 2 and 3 ran on battery, excluded
        try:
            times[row["test"]].append(float(row["seconds"]))
        except ValueError:
            pass  # a run whose time could not be read is skipped


def mean(k):
    return st.mean(times[k]) if times[k] else float("nan")


def sd(k):
    return st.stdev(times[k]) if len(times[k]) > 1 else 0.0


lines = []
out = lines.append

out(f"{'test':8s} {'n':>2s} {'mean (s)':>10s} {'std (s)':>9s}")
for k in ["ner_seq", "ner_t1", "ner_t8", "re_t1", "re_t8",
          "er_cpu", "er_cuda", "mpi_p1", "mpi_p8"]:
    if times[k]:
        out(f"{k:8s} {len(times[k]):2d} {mean(k):10.2f} {sd(k):9.2f}")
    else:
        out(f"{k:8s}  0 (no data)")

out("")
out("Stage comparisons (means)")
out(f"{'comparison':32s} {'speedup':>8s} {'workers':>8s} {'efficiency':>11s} {'worker-s':>9s}")


def row(name, base, par, workers):
    if not (times[base] and times[par]):
        return
    sp = mean(base) / mean(par)
    eff = sp / workers
    ws = mean(par) * workers
    out(f"{name:32s} {sp:7.2f}x {workers:8d} {eff*100:10.1f}% {ws:9.0f}")


row("NER: 1 thread -> 8 threads", "ner_t1", "ner_t8", 8)
row("NER: sequential script -> 8 thr", "ner_seq", "ner_t8", 8)
row("RE : 1 thread -> 8 threads", "re_t1", "re_t8", 8)
row("ER : CPU -> CUDA (1 GPU)", "er_cpu", "er_cuda", 1)
row("MPI: 1 process -> 8 processes", "mpi_p1", "mpi_p8", 8)

text = "\n".join(lines)
print(text)
with open("trial_summary.txt", "w") as f:
    f.write(text + "\n")
print("\nSaved trial_summary.txt")
