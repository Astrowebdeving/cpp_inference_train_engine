#!/bin/zsh
# usage: detbench.sh LABEL BINARY [bench args...] -> three runs, compares exported final weights
label=$1; bin=$2; shift 2
D=/private/tmp/claude-504/-Users-tu15-alex-cpp/31e524c9-a66a-4a3a-8be5-706d442a7ae5/scratchpad/verify/detb
mkdir -p $D; shas=()
for i in 1 2 3; do
  rm -rf $D/$label-$i
  $bin --kind train --backend mps --attention reference --grad-accum 1 --submission-window 4 --warmup 0 --repetitions 6 --skip-cpu-check --comparison-data $D/$label-$i "$@" > $D/$label-$i.log 2>&1 || { echo "$label: FAILED"; tail -2 $D/$label-$i.log; exit 1; }
  shas+=($(shasum -a 256 $D/$label-$i/tensors.bin | cut -c1-12))
done
if [[ ${shas[1]} == ${shas[2]} && ${shas[2]} == ${shas[3]} ]]; then verdict="DETERMINISTIC"; else verdict="varies"; fi
printf "%-40s %-14s %s\n" "$label" "$verdict" "${shas[*]}"
