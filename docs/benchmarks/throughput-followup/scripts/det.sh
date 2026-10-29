#!/bin/zsh
# usage: det.sh LABEL BINARY CONFIG [extra train args...]  -> runs twice, prints whether weights are identical
label=$1; bin=$2; cfg=$3; shift 3
V=/private/tmp/claude-504/-Users-tu15-alex-cpp/31e524c9-a66a-4a3a-8be5-706d442a7ae5/scratchpad/verify/det
shas=()
for i in 1 2 3; do
  rm -rf $V/$label-$i
  $bin --config $cfg --output $V/$label-$i "$@" > $V/$label-$i.log 2>&1 || { echo "$label: FAILED"; tail -2 $V/$label-$i.log; exit 1; }
  shas+=($(shasum -a 256 $V/$label-$i/checkpoint-latest/weights.bin | cut -c1-12))
done
if [[ ${shas[1]} == ${shas[2]} && ${shas[2]} == ${shas[3]} ]]; then verdict="DETERMINISTIC"; else verdict="varies"; fi
printf "%-34s %-14s %s\n" "$label" "$verdict" "${shas[*]}"
