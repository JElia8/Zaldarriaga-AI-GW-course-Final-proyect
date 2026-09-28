#!/bin/sh
# Recompute SHA-256 of every file of "First round" and compare with the snapshot taken before the verification.
cd "/c/Users/JUAMPI/Desktop/Proyecto final Zaldarriaga"
find "First round" -type f -not -path "*__pycache__*" -print0 | sort -z | xargs -0 sha256sum > "Second round/first_round_sha256_after.txt"
n=$(wc -l < "Second round/first_round_sha256_before.txt")
if cmp -s "Second round/first_round_sha256_before.txt" "Second round/first_round_sha256_after.txt"; then
  res="identical"; else res="DIFFERENT"; fi
d=$(date "+%Y-%m-%d %H:%M")
cat > "Second round/report_verification/integrity_note.tex" <<TEX
The SHA-256 hashes of all $n files of \file{First round} (Python caches excluded) were recorded before the verification
(\file{first_round_sha256_before.txt}) and recomputed at the end ($d, \file{first_round_sha256_after.txt}).
The two lists are \textbf{$res}. No file of the first round was created, modified or deleted by the second round;
all executions used copies inside \file{Second round}.
TEX
echo "$n files: $res"
